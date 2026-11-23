// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-11-23

// pySMESH binding — the thread every SMESH and NETGEN compute runs on (report #1).
//
// A compute ran on the caller's thread. The main thread of python.exe reserves 2 MB of
// stack, and a NETGEN 1D-2D compute of a 10-face model needs more than 2 MB and at most
// 4 MB: it ended the process with 0xC00000FD (stack overflow) on the main thread and
// passed on any thread of 4 MB or more. pySMESH chooses the thread, so it runs each
// compute on a thread of its own with a 256 MB stack reservation, and joins it. The
// reservation is address space; Windows commits the stack page by page as it grows.
//
// std::thread cannot set a stack size, so the thread comes from _beginthreadex with
// STACK_SIZE_PARAM_IS_A_RESERVATION.
//
// What the worker takes over from the caller, each checked at the source:
//   * the floating-point control word (_controlfp_s: exception masks, rounding, denormal
//     control). It is per thread, so the caller's is read and set on the worker.
//   * nothing else. pySMESH sets no SEH translator (_set_se_translator) on any thread, and
//     OCCT's OSD::SetSignal is not called: on MSVC OCC_CATCH_SIGNALS is empty
//     (Standard_ErrorHandler.hxx) and SMESH's only call is in SMESH_Gen_i, which pySMESH
//     does not build. OCCT's Standard_ErrorHandler keeps one handler stack per thread and
//     makes it on first use. NETGEN's only thread-local is TaskManager::thread_id
//     (taskmanager.cpp), 0 on a new thread as on the caller's, and its RegionTaskManager
//     enters and leaves its task manager inside the call. NETGEN's globals are process
//     globals, which the caller's netgen lock still guards. Locale is per process.
//
// A C++ exception on the worker is caught there and re-thrown on the caller, so it
// reaches Python as it did before. A structured exception (an access violation) is not
// caught on either thread: it ends the process with its own code, as it did on the
// caller's thread. The work runs with the GIL released and must not call into Python;
// the progress and cancel hooks keep their own helper thread.

#pragma once

#include <functional>

namespace pysmesh {

// The stack reservation of the compute thread: 256 MB of address space.
constexpr unsigned kComputeStackReserve = 256u * 1024u * 1024u;

// Run `work` on a new thread with a kComputeStackReserve stack and wait for it. The
// caller's floating-point control word is set on the thread first. An exception `work`
// throws is re-thrown here. Throws PysmeshError if the thread cannot be started. Call it
// with the GIL released.
void run_on_compute_thread(const std::function<void()>& work);

}  // namespace pysmesh
