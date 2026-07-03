/*
 * Minimal pthreads compatibility header for MSVC on Windows.
 * Maps pthread primitives to native Windows APIs (SRWLOCK, CreateThread).
 * Only implements the subset used by the SALOME Kernel code.
 */

#ifndef _PTHREAD_COMPAT_H
#define _PTHREAD_COMPAT_H

#ifdef _MSC_VER

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
/* SRW locks require Vista+ (_WIN32_WINNT >= 0x0600) */
#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif
#include <windows.h>
#include <process.h>
#include <stdlib.h>

typedef SRWLOCK pthread_mutex_t;
typedef void*   pthread_mutexattr_t;
typedef DWORD   pthread_t;
typedef void*   pthread_attr_t;

#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT

static inline int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) {
    (void)a;
    InitializeSRWLock(m);
    return 0;
}

static inline int pthread_mutex_destroy(pthread_mutex_t *m) {
    (void)m;
    return 0;
}

static inline int pthread_mutex_lock(pthread_mutex_t *m) {
    AcquireSRWLockExclusive(m);
    return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *m) {
    ReleaseSRWLockExclusive(m);
    return 0;
}

static inline pthread_t pthread_self(void) {
    return GetCurrentThreadId();
}

typedef struct {
    void *(*routine)(void *);
    void *arg;
} _pthread_start_info;

static unsigned __stdcall _pthread_start_wrapper(void *data) {
    _pthread_start_info *info = (_pthread_start_info *)data;
    info->routine(info->arg);
    free(info);
    return 0;
}

static inline int pthread_create(pthread_t *tid, const pthread_attr_t *attr,
                                 void *(*start_routine)(void *), void *arg) {
    (void)attr;
    _pthread_start_info *info = (_pthread_start_info *)malloc(sizeof(_pthread_start_info));
    if (!info) return -1;
    info->routine = start_routine;
    info->arg = arg;
    HANDLE h = (HANDLE)_beginthreadex(NULL, 0, _pthread_start_wrapper, info, 0, (unsigned *)tid);
    if (h == 0) { free(info); return -1; }
    CloseHandle(h);
    return 0;
}

static inline int pthread_join(pthread_t tid, void **retval) {
    (void)retval;
    HANDLE h = OpenThread(SYNCHRONIZE, FALSE, tid);
    if (h) {
        WaitForSingleObject(h, INFINITE);
        CloseHandle(h);
    }
    return 0;
}

static inline void pthread_exit(void *retval) {
    (void)retval;
    _endthreadex(0);
}

#endif /* _MSC_VER */
#endif /* _PTHREAD_COMPAT_H */
