#ifndef CRASHHANDLER_HPP
#define CRASHHANDLER_HPP

#include <dlfcn.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>
#include <unwind.h>

#include <cinttypes>
#include <cstdio>
#include <initializer_list>

// A native backtrace on the way out of a fatal signal.
//
// A SIGSEGV in this process produces nothing the app can show. The backend is
// a separate executable, so there is no Java crash dialog; its stdout is a pipe
// that dies with it; and the tombstone lands in /data/tombstones, which an
// unrooted phone will not surrender. What reaches the user is "exited with code
// 139" and no location -- which is where one failure sat for several rounds,
// with the search narrowed only by which log line came last.
//
// So the handler writes the signal, the faulting address and an unwound stack
// straight to stdout, which the app already drains into the log panel it shows.
//
// dladdr and snprintf are not async-signal-safe, and this calls both. That is
// a deliberate trade: the process is already dying on a fatal signal, the worst
// case is a truncated or missing frame, and an unsymbolised list of hex
// addresses from a stripped .so would not have answered the question anyway.
// The handler re-raises with the default disposition afterwards so the exit
// status the app reports stays the real one.
namespace crash_handler {

struct Frames {
  void *pc[64];
  int count = 0;
};

inline _Unwind_Reason_Code collect(struct _Unwind_Context *ctx, void *arg) {
  auto *f = static_cast<Frames *>(arg);
  uintptr_t ip = _Unwind_GetIP(ctx);
  if (ip == 0) return _URC_END_OF_STACK;
  if (f->count >= (int)(sizeof(f->pc) / sizeof(f->pc[0])))
    return _URC_END_OF_STACK;
  f->pc[f->count++] = reinterpret_cast<void *>(ip);
  return _URC_NO_REASON;
}

inline void emit(const char *s) {
  ssize_t n = write(STDOUT_FILENO, s, strlen(s));
  (void)n;
}

inline void onFatal(int sig, siginfo_t *info, void *) {
  char line[512];
  const char *name = sig == SIGSEGV   ? "SIGSEGV"
                     : sig == SIGBUS  ? "SIGBUS"
                     : sig == SIGABRT ? "SIGABRT"
                     : sig == SIGILL  ? "SIGILL"
                     : sig == SIGFPE  ? "SIGFPE"
                                      : "signal";
  snprintf(line, sizeof(line), "\n*** CRASH %s at address %p ***\n", name,
           info ? info->si_addr : nullptr);
  emit(line);

  Frames f;
  _Unwind_Backtrace(&collect, &f);
  // Frame 0 is this handler; the interesting ones start a little above it, but
  // print everything rather than guess at the trampoline depth.
  for (int i = 0; i < f.count; ++i) {
    Dl_info dli;
    // The module-relative offset is printed whether or not a name resolves.
    // Address-space layout is randomised, so the absolute pc means nothing off
    // the device; pc - dli_fbase is a file offset, and that is what the
    // unstripped artifact under build/android/bin/ can be symbolised against
    // when a frame is inlined too deeply for dladdr to name it.
    const bool known = dladdr(f.pc[i], &dli) != 0;
    const char *fname = known && dli.dli_fname ? dli.dli_fname : nullptr;
    const char *base = fname ? strrchr(fname, '/') : nullptr;
    const char *mod = base ? base + 1 : (fname ? fname : "?");
    if (known && dli.dli_sname) {
      snprintf(line, sizeof(line), "  #%02d %s+0x%tx  %s+0x%tx\n", i, mod,
               (char *)f.pc[i] - (char *)dli.dli_fbase, dli.dli_sname,
               (char *)f.pc[i] - (char *)dli.dli_saddr);
    } else if (known) {
      snprintf(line, sizeof(line), "  #%02d %s+0x%tx\n", i, mod,
               (char *)f.pc[i] - (char *)dli.dli_fbase);
    } else {
      snprintf(line, sizeof(line), "  #%02d %p (no module)\n", i, f.pc[i]);
    }
    emit(line);
  }
  emit("*** end of backtrace ***\n");

  // Restore the default and re-raise, so the process still dies of what killed
  // it and the app reports the true exit status.
  signal(sig, SIG_DFL);
  raise(sig);
}

inline void install() {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = &onFatal;
  sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
  sigemptyset(&sa.sa_mask);
  for (int sig : {SIGSEGV, SIGBUS, SIGABRT, SIGILL, SIGFPE})
    sigaction(sig, &sa, nullptr);
}

}  // namespace crash_handler

#endif  // CRASHHANDLER_HPP
