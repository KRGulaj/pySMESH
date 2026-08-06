// pySMESH binding — progress reporting and cancellation for the long session operations.
//
// OCCT's progress mechanism is a Message_ProgressIndicator handed to an algorithm as a
// Message_ProgressRange. Two of its methods are the whole interface, and both are on the
// algorithm's hot path:
//
//   Show()      — called on every advance of the position;
//   UserBreak() — called to ask whether the caller wants to stop.
//
// Measured on this project's 117-solid assembly, one boolean called Show() 291 303 times and
// UserBreak() 908 395 times; a two-box fuse that finishes in 4 ms still calls Show() 292
// times. Both may be called from OCCT's own worker threads when the algorithm runs in
// parallel. Calling a Python callable from either would therefore mean acquiring the GIL
// hundreds of thousands of times from inside a parallel algorithm — which would cost far
// more than the operation and would serialise the very thing the parallel flag exists to
// speed up. OCCT's own documentation says as much: UserBreak() "should return as soon as
// possible", and Show() "should return as soon as possible to reduce thread contention".
//
// So neither hook talks to Python. Show() stores a double, UserBreak() reads a bool, and a
// separate polling thread does the talking while the operation runs with the GIL released.
// That thread is the only place Python is entered, it enters at a bounded rate, and it is
// joined before the operation returns.
//
// See session/session.hpp for the file split.

#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <string>
#include <thread>

#include <Message_ProgressIndicator.hxx>
#include <Message_ProgressRange.hxx>
#include <Message_ProgressScope.hxx>
#include <Standard_Handle.hxx>

#include "../common.hpp"

namespace pysmesh {
namespace session {

// The two Python callables an operation may be driven by, plus how often to consult them.
//
// Two callables rather than one returning a bool: a progress bar and a cancel button are
// separate concerns, and a caller that wants only one should not have to supply the other.
// Both members default to Py_None rather than to a default-constructed py::object, which
// would be NULL — and a NULL py::object's is_none() is false, so active() would report every
// hookless operation as hooked. Constructing one therefore needs the GIL, which every call
// site holds.
struct ProgressHooks {
  py::object on_progress = py::none();    // Callable[[float], None], or None
  py::object should_cancel = py::none();  // Callable[[], bool], or None
  double interval_s = 0.025;

  bool active() const { return !on_progress.is_none() || !should_cancel.is_none(); }
};

// A progress indicator whose two hot methods touch nothing but an atomic.
//
// Deliberately carries no OCCT RTTI macros: it is never dynamically cast, and the macros
// would need an out-of-line IMPLEMENT_STANDARD_RTTIEXT for no benefit.
class AtomicProgress : public Message_ProgressIndicator {
 public:
  // Called by OCCT, possibly concurrently, possibly on a worker thread. Must be trivial.
  bool UserBreak() override { return cancelled_.load(std::memory_order_relaxed); }

  // Called by OCCT under its own mutex on every advance. GetPosition() is documented as safe
  // to read from exactly here.
  void Show(const Message_ProgressScope&, bool) override {
    position_.store(GetPosition(), std::memory_order_relaxed);
  }

  double position() const { return position_.load(std::memory_order_relaxed); }
  void cancel() { cancelled_.store(true, std::memory_order_relaxed); }
  bool cancelled() const { return cancelled_.load(std::memory_order_relaxed); }

 private:
  std::atomic<double> position_{0.0};
  std::atomic<bool> cancelled_{false};
};

// Runs the Python hooks on a helper thread for the duration of one operation.
//
// Lifetime is scoped to the operation: construct it under the GIL before releasing it, hand
// range() to the algorithm, and call finish() after the GIL is back. The destructor stops
// and joins unconditionally, so an exception thrown out of the algorithm cannot leave the
// thread running against a dead indicator.
class ProgressDriver {
 public:
  // `op` names the operation in any error this raises. Nothing is started when neither hook
  // is given, and range() then returns OCCT's own inert range — an operation driven without
  // hooks pays nothing at all.
  ProgressDriver(const char* op, const ProgressHooks& hooks);

  ~ProgressDriver();

  ProgressDriver(const ProgressDriver&) = delete;
  ProgressDriver& operator=(const ProgressDriver&) = delete;

  // The range to hand the algorithm. An inert range when no hook was supplied.
  Message_ProgressRange range();

  // Stop the poller, deliver the final position, and re-raise whatever a hook threw.
  //
  // Must be called with the GIL held, after the algorithm has returned. A hook's exception
  // is raised here rather than where it happened, because where it happened is a thread with
  // no Python frame to raise into — and it also cancelled the operation, so the algorithm
  // has already stopped by the time this runs.
  void finish();

  // True when a cancel was requested — by the predicate, or by a hook raising.
  //
  // This, and not the algorithm's own reporting, is what an operation must test. A cancelled
  // ShapeFix_Shape returns a non-null shape carrying 436 of 5606 faces (measured), so
  // trusting the algorithm would commit a truncated model as a success.
  bool cancelled() const;

  // Raise the cancellation error for `op`. Every operation words it the same way, because
  // the caller's handling is the same in every case: nothing happened.
  [[noreturn]] static void raise_cancelled(const char* op);

 private:
  // The poller's loop, run on the helper thread.
  void poll();

  // Signal the poller to stop and join it. Idempotent. Releases the GIL around the join when
  // this thread holds it, because the poller may be blocked acquiring it for one last tick.
  void stop_thread();

  const char* op_;
  ProgressHooks hooks_;
  Handle(AtomicProgress) indicator_;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable wake_;
  bool stop_ = false;
  bool finished_ = false;
  double last_reported_ = -1.0;

  // What a hook raised, re-thrown by finish() so the caller sees their own exception with
  // its own type and traceback rather than a stringified copy of it. It owns Python objects,
  // so it is only ever assigned and cleared with the GIL held.
  std::exception_ptr hook_error_;
};

}  // namespace session
}  // namespace pysmesh
