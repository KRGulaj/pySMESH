// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-08-05

// pySMESH v2 capability probe — shared check harness.
//
// The probe is a build-verification target, not a unit-test suite: it exists to prove that
// every capability the v2 Tier-C modelling surface needs is (a) declared by a header we ship
// against, (b) resolvable at link time out of the OCCT toolkits and SMESH static libraries
// the wheel links, and (c) callable at run time in this exact binary. Anything it cannot do
// is a build/patching problem and must be fixed before the v2 bindings are written; anything
// it can do is a pure coding task from there on.
//
// Numeric assertions are deliberate but coarse — this is not where the v2 acceptance gates
// live (those belong in pytest, against the real bindings). A value is checked only where a
// wrong value would mean the library is mis-linked or mis-configured rather than merely
// mis-parameterised.

#ifndef PYSMESH_TESTS_PROBE_PROBE_HPP
#define PYSMESH_TESTS_PROBE_PROBE_HPP

#include <string>

namespace probe {

// Start a named capability block. Printed once, before its checks.
void section(const std::string& id, const std::string& title);

// Record one boolean check. `label` is printed on failure and must name the requirement.
void check(bool ok, const std::string& label);

// Record one numeric check against a closed-form/reference value with absolute tolerance.
void check_close(double got, double want, double tol, const std::string& label);

// Record a capability that is reachable but deliberately not exercised numerically here,
// with the reason. Counts as neither pass nor fail; reported separately so the report can
// state exactly what the probe did and did not run.
void note(const std::string& label, const std::string& reason);

int checks_run();
int checks_failed();

// Print the summary block. Returns the process exit code (0 == every check passed).
int summarize();

}  // namespace probe

// Implemented in probe_occt.cpp / probe_smesh.cpp.
void run_occt_probe();
void run_smesh_probe();

#endif  // PYSMESH_TESTS_PROBE_PROBE_HPP
