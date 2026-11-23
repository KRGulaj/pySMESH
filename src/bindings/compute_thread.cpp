// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-11-23

// pySMESH binding — the thread every SMESH and NETGEN compute runs on. See
// compute_thread.hpp for why, and for what the worker takes over from the caller.

#include "compute_thread.hpp"

#include "common.hpp"

#include <cerrno>
#include <cstring>
#include <exception>
#include <string>

#include <float.h>
#include <process.h>

#include <windows.h>

namespace pysmesh {
namespace {

// What the caller hands the worker, and what the worker hands back.
struct ComputeJob {
  const std::function<void()>* work = nullptr;
  unsigned int control_word = 0;
  std::exception_ptr error;
};

// The control word bits that _controlfp_s may set on x64: exception masks, rounding
// and denormal control. _MCW_PC (precision) is not supported there.
constexpr unsigned int kControlMask = _MCW_EM | _MCW_RC | _MCW_DN;

unsigned __stdcall compute_thread_main(void* arg) {
  ComputeJob& job = *static_cast<ComputeJob*>(arg);
  unsigned int unused = 0;
  _controlfp_s(&unused, job.control_word, kControlMask);
  try {
    (*job.work)();
  } catch (...) {
    job.error = std::current_exception();
  }
  return 0;
}

}  // namespace

void run_on_compute_thread(const std::function<void()>& work) {
  ComputeJob job;
  job.work = &work;
  _controlfp_s(&job.control_word, 0, 0);
  const uintptr_t handle =
      _beginthreadex(nullptr, kComputeStackReserve, &compute_thread_main, &job,
                     STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
  if (handle == 0) {
    const int code = errno;
    throw PysmeshError("pySMESH: the compute thread (" +
                       std::to_string(kComputeStackReserve / (1024u * 1024u)) +
                       " MB stack reservation) could not be started: " +
                       std::string(std::strerror(code)) + " (errno " +
                       std::to_string(code) + ").");
  }
  WaitForSingleObject(reinterpret_cast<HANDLE>(handle), INFINITE);
  CloseHandle(reinterpret_cast<HANDLE>(handle));
  if (job.error) {
    std::rethrow_exception(job.error);
  }
}

}  // namespace pysmesh
