// pySMESH v2 capability probe — runner and check harness implementation.
//
// Build:  cmake -DPYSMESH_BUILD_V2_PROBE=ON ... && cmake --build build --target v2_probe
// Run:    build/v2_probe.exe        (exit 0 == every capability probed is usable)

#include "probe.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace probe {
namespace {

int g_run = 0;
int g_failed = 0;
std::vector<std::string> g_failures;
std::vector<std::string> g_notes;

}  // namespace

void section(const std::string& id, const std::string& title) {
  std::printf("\n=== %s — %s\n", id.c_str(), title.c_str());
}

void check(bool ok, const std::string& label) {
  ++g_run;
  if (ok) {
    std::printf("  [ ok ] %s\n", label.c_str());
    return;
  }
  ++g_failed;
  g_failures.push_back(label);
  std::printf("  [FAIL] %s\n", label.c_str());
}

void check_close(double got, double want, double tol, const std::string& label) {
  const bool ok = std::isfinite(got) && std::fabs(got - want) <= tol;
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s (got %.10g, want %.10g +/- %.3g)", label.c_str(), got,
                want, tol);
  check(ok, buf);
}

void note(const std::string& label, const std::string& reason) {
  g_notes.push_back(label + " — " + reason);
  std::printf("  [note] %s (%s)\n", label.c_str(), reason.c_str());
}

int checks_run() { return g_run; }
int checks_failed() { return g_failed; }

int summarize() {
  std::printf("\n================ v2 capability probe summary ================\n");
  std::printf("checks run    : %d\n", g_run);
  std::printf("checks failed : %d\n", g_failed);
  if (!g_notes.empty()) {
    std::printf("notes (%zu):\n", g_notes.size());
    for (const std::string& n : g_notes) {
      std::printf("  - %s\n", n.c_str());
    }
  }
  if (g_failed != 0) {
    std::printf("FAILURES:\n");
    for (const std::string& f : g_failures) {
      std::printf("  - %s\n", f.c_str());
    }
    std::printf("RESULT: FAIL\n");
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}

}  // namespace probe

int main() {
  std::printf("pySMESH v2 capability probe\n");
  run_occt_probe();
  run_smesh_probe();
  return probe::summarize();
}
