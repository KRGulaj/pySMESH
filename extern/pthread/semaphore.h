/*
 * Minimal POSIX semaphore compatibility header for MSVC on Windows.
 * Maps sem_* primitives to native Windows semaphore APIs.
 * Only implements the subset used by the SALOME Kernel code.
 */

#ifndef _SEMAPHORE_COMPAT_H
#define _SEMAPHORE_COMPAT_H

#ifdef _MSC_VER

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef HANDLE sem_t;

static inline int sem_init(sem_t *sem, int pshared, unsigned int value) {
    (void)pshared;
    *sem = CreateSemaphoreA(NULL, (LONG)value, 0x7FFFFFFF, NULL);
    return (*sem == NULL) ? -1 : 0;
}

static inline int sem_wait(sem_t *sem) {
    return (WaitForSingleObject(*sem, INFINITE) == WAIT_OBJECT_0) ? 0 : -1;
}

static inline int sem_post(sem_t *sem) {
    return ReleaseSemaphore(*sem, 1, NULL) ? 0 : -1;
}

static inline int sem_destroy(sem_t *sem) {
    return CloseHandle(*sem) ? 0 : -1;
}

static inline int sem_getvalue(sem_t *sem, int *sval) {
    (void)sem;
    *sval = 0;
    return 0;
}

#endif /* _MSC_VER */
#endif /* _SEMAPHORE_COMPAT_H */
