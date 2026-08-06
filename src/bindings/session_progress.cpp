// pySMESH binding — the progress/cancellation driver.
//
// See session/progress.hpp for why the Python hooks are polled from a helper thread rather
// than called from OCCT's own Show()/UserBreak().
//
// See session/session.hpp for the file split.

#include "session/progress.hpp"

#include <chrono>
#include <utility>

namespace pysmesh {
namespace session {

ProgressDriver::ProgressDriver(const char* op, const ProgressHooks& hooks)
    : op_(op), hooks_(hooks) {
  if (!hooks_.active()) {
    return;
  }
  if (!(hooks_.interval_s > 0.0)) {
    throw PysmeshError(std::string("Session.") + op +
                       ": the progress poll interval must be > 0 s (got " +
                       std::to_string(hooks_.interval_s) + ").");
  }
  indicator_ = new AtomicProgress;

  // Ask once, here, before anything starts. Without this the contract would be a race with
  // the clock: an operation that finishes inside one poll interval — a two-box fuse takes
  // 4 ms against a 25 ms interval — would run to completion however emphatically the
  // predicate said no, so a caller who set their cancel flag *before* calling would be
  // ignored precisely on the cheap operations. Asking first makes "already cancelled" mean
  // the same thing at every size. A predicate that raises here propagates straight out of
  // the caller's own call, which is the clearest place for it to surface.
  if (!hooks_.should_cancel.is_none() && hooks_.should_cancel().cast<bool>()) {
    indicator_->cancel();
    return;  // no poller: there is nothing left to ask.
  }
  worker_ = std::thread([this] { poll(); });
}

ProgressDriver::~ProgressDriver() {
  stop_thread();
  if (hook_error_) {
    // The stored exception owns Python objects, so it must be released under the GIL.
    py::gil_scoped_acquire acquire;
    hook_error_ = nullptr;
  }
}

Message_ProgressRange ProgressDriver::range() {
  if (indicator_.IsNull()) {
    // OCCT's own inert range. An algorithm handed one reports nothing and is never asked to
    // break, which is exactly what an operation with no hooks should cost.
    return Message_ProgressRange();
  }
  return indicator_->Start();
}

bool ProgressDriver::cancelled() const {
  return !indicator_.IsNull() && indicator_->cancelled();
}

void ProgressDriver::raise_cancelled(const char* op) {
  throw CancelledError(std::string("Session.") + op + ": cancelled by the caller.",
                       "The session is unchanged: no id was issued, the operation counter "
                       "did not advance, and no partial shape was committed.");
}

void ProgressDriver::finish() {
  if (finished_) {
    return;
  }
  finished_ = true;
  stop_thread();

  // A hook that raised is re-raised here, on the thread that has a Python frame to raise
  // into. It also cancelled the operation, so the algorithm has already stopped.
  if (hook_error_) {
    const std::exception_ptr raised = hook_error_;
    hook_error_ = nullptr;
    std::rethrow_exception(raised);
  }

  // A progress bar has to reach the end, and the poller cannot deliver the last position
  // because the operation finishes between two of its ticks. Only for an operation that ran
  // to completion: reporting 1.0 for a cancelled one would be a lie.
  if (indicator_.IsNull() || indicator_->cancelled() || hooks_.on_progress.is_none()) {
    return;
  }
  const double p = indicator_->position();
  if (p > last_reported_) {
    last_reported_ = p;
    hooks_.on_progress(p);
  }
}

void ProgressDriver::stop_thread() {
  if (!worker_.joinable()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  // The poller may be blocked acquiring the GIL for a tick. Joining while holding the GIL
  // would deadlock the two against each other, so it is released for the join — but only if
  // this thread actually holds it, because the destructor can run on either path.
  if (PyGILState_Check()) {
    py::gil_scoped_release release;
    worker_.join();
  } else {
    worker_.join();
  }
}

void ProgressDriver::poll() {
  for (;;) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_.wait_for(lock, std::chrono::duration<double>(hooks_.interval_s),
                     [this] { return stop_; });
      if (stop_) {
        return;
      }
    }

    py::gil_scoped_acquire acquire;
    try {
      // Report first, then ask. A caller that cancels in response to the position it was
      // just shown has therefore seen that position.
      const double p = indicator_->position();
      if (!hooks_.on_progress.is_none() && p > last_reported_) {
        last_reported_ = p;
        hooks_.on_progress(p);
      }
      if (!hooks_.should_cancel.is_none() && hooks_.should_cancel().cast<bool>()) {
        indicator_->cancel();
        return;
      }
    } catch (...) {
      // A hook that raises is a cancel: the operation stops, nothing is committed, and the
      // exception reaches the caller from finish(). Swallowing it and carrying on would run
      // a long operation to completion that the caller's own code had already given up on.
      hook_error_ = std::current_exception();
      indicator_->cancel();
      return;
    }
  }
}

}  // namespace session
}  // namespace pysmesh
