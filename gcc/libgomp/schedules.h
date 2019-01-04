
#ifndef SCHEDULES_H
#define SCHEDULES_H 1

#include <math.h>
#include <stdio.h>
#include "libgomp.h"

typedef unsigned long long gomp_ull;

inline gomp_ull
clip_ull(gomp_ull x,
     gomp_ull low,
     gomp_ull hi)
{
    gomp_ull temp = x;
    if(x < low)
        temp = low;
    else if(x > hi)
        temp = hi;
    return temp;
}

inline long
clip(long x, long low, long hi)
{
    long temp = x;
    if(x < low)
        temp = low;
    else if(x > hi)
        temp = hi;
    return temp;
}


inline gomp_ull
css_chunk_size_ull(double param, gomp_ull N, size_t P)
{
    gomp_ull chunk_size =
        pow((1.414213562 * N * param) / (P * sqrt(log(P))), 2/3);
    return clip_ull(chunk_size, 1, N);
}

inline long
css_chunk_size(double param, long N, size_t P)
{
    long chunk_size =
        pow((1.414213562 * N * param) / (P * sqrt(log(P))), 2/3);
    return clip(chunk_size, 1, N);
}

inline double
fss_transform_range(double param)
{
    return param;
}

#ifdef HAVE_SYNC_BUILTINS
/* Similar, but doesn't require the lock held, and uses compare-and-swap
   instead.  Note that the only memory value that changes is ws->next.  */

typedef unsigned long bo_ul;

inline static bool
bo_iter_fac2_next (long *pstart, long *pend)
{
    struct gomp_thread *thr = gomp_thread ();
    struct gomp_work_share *ws = thr->ts.work_share;
    struct gomp_team *team = thr->ts.team;
    bo_ul nthreads = team ? team->nthreads : 1;
    long start = __atomic_load_n (&ws->next, MEMMODEL_RELAXED);
    long end = ws->end;
    long incr = ws->incr;
    long barrier = __atomic_load_n (&ws->barrier, MEMMODEL_RELAXED);
    bo_ul chunk_size = __atomic_load_n (&ws->chunk_size, MEMMODEL_RELAXED);
    bo_ul R = (end - start) / incr;
    bo_ul nend;

    while (1)
    {
        if (start == end)
            return false;

        R = (end - start) / incr;

        if(barrier == start)
        {
            bo_ul F = (R / 2) / nthreads;
            bo_ul PF  = F * nthreads;
            bo_ul nbarrier = barrier + (PF * incr);

            if(F <= 1)
            {
                F = 1;
                nbarrier = end;
            }

            bo_ul chunk_tmp =
                __sync_val_compare_and_swap (&ws->chunk_size,
                                             chunk_size,
                                             F);
            if (chunk_tmp != chunk_size) {
                chunk_size = chunk_tmp;
                start = __atomic_load_n (&ws->next, MEMMODEL_RELAXED);
                barrier = __atomic_load_n (&ws->barrier, MEMMODEL_RELAXED);
                continue;
            }
            else {
                chunk_size = F;
            }

            bo_ul barrier_tmp =
                __sync_val_compare_and_swap (&ws->barrier,
                                             barrier,
                                             nbarrier);

            if (barrier_tmp != barrier)
            {
                barrier = barrier_tmp;
                start = __atomic_load_n (&ws->next, MEMMODEL_RELAXED);
                continue;
            }
            else {
                barrier = nbarrier;
            }
        }

        if(__builtin_expect (chunk_size <= R, 1))
            nend = start + chunk_size * incr;
        else
            nend = end;

        long tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
        if (tmp == start)
            break;

        start = tmp;
        barrier = __atomic_load_n (&ws->barrier, MEMMODEL_RELAXED);
        chunk_size = __atomic_load_n (&ws->chunk_size, MEMMODEL_RELAXED);
    }

    *pstart = start;
    *pend = nend;
    return true;
}

inline static bool
bo_iter_ull_fac2_next (gomp_ull *pstart, gomp_ull *pend)
{
    struct gomp_thread *thr = gomp_thread ();
    struct gomp_work_share *ws = thr->ts.work_share;
    struct gomp_team *team = thr->ts.team;
    gomp_ull nthreads = team ? team->nthreads : 1;
    gomp_ull start = __atomic_load_n (&ws->next_ull, MEMMODEL_RELAXED);
    gomp_ull end = ws->end_ull;
    gomp_ull incr = ws->incr_ull;
    gomp_ull barrier = __atomic_load_n (&ws->barrier_ull, MEMMODEL_RELAXED);
    gomp_ull chunk_size = __atomic_load_n (&ws->chunk_size_ull, MEMMODEL_RELAXED);
    gomp_ull nend;
    gomp_ull R;

    while (1)
    {
        if (start == end)
            return false;

        if (__builtin_expect (ws->mode, 0) == 0)
            R = (end - start) / incr;
        else
            R = (start - end) / -incr;

        if(barrier == start)
        {
            gomp_ull F = (R / 2) / nthreads;
            gomp_ull PF  = F * nthreads;
            gomp_ull nbarrier;

            if (__builtin_expect (ws->mode, 0) == 0)
                nbarrier = barrier + (PF * incr);
            else
                nbarrier = barrier + (PF * -incr);

            if(F <= 1)
            {
                F = 1;
                nbarrier = end;
            }

            gomp_ull chunk_tmp =
                __sync_val_compare_and_swap (&ws->chunk_size_ull,
                                             chunk_size,
                                             F);
            if (chunk_tmp != chunk_size) {
                chunk_size = chunk_tmp;
                start = __atomic_load_n (&ws->next_ull, MEMMODEL_RELAXED);
                barrier = __atomic_load_n (&ws->barrier_ull, MEMMODEL_RELAXED);
                continue;
            }
            else {
                chunk_size = F;
            }

            gomp_ull barrier_tmp =
                __sync_val_compare_and_swap (&ws->barrier_ull,
                                             barrier,
                                             nbarrier);

            if (barrier_tmp != barrier)
            {
                barrier = barrier_tmp;
                start = __atomic_load_n (&ws->next_ull, MEMMODEL_RELAXED);
                continue;
            }
            else {
                barrier = nbarrier;
            }
        }
        
        if(__builtin_expect (chunk_size <= R, 1))
        {
            if (__builtin_expect (ws->mode, 0) == 0)
                nend = start + chunk_size * incr;
            else
                nend = start + chunk_size * -incr;
        }
        else
            nend = end;

        gomp_ull tmp = __sync_val_compare_and_swap (&ws->next_ull, start, nend);
        if (tmp == start)
            break;

        start = tmp;
        barrier = __atomic_load_n (&ws->barrier_ull, MEMMODEL_RELAXED);
        chunk_size = __atomic_load_n (&ws->chunk_size_ull, MEMMODEL_RELAXED);
    }

    *pstart = start;
    *pend = nend;
    return true;
}

inline static bool
bo_iter_fss_next (long *pstart, long *pend)
{
    struct gomp_thread *thr = gomp_thread ();
    struct gomp_work_share *ws = thr->ts.work_share;
    struct gomp_team *team = thr->ts.team;
    bo_ul nthreads = team ? team->nthreads : 1;
    long start = __atomic_load_n (&ws->next, MEMMODEL_RELAXED);
    long end = ws->end;
    long incr = ws->incr;
    long barrier = __atomic_load_n (&ws->barrier, MEMMODEL_RELAXED);
    bo_ul chunk_size = __atomic_load_n (&ws->chunk_size, MEMMODEL_RELAXED);
    bo_ul R = (end - start) / incr;
    bo_ul nend;

    double param = fss_transform_range(ws->param);

    while (1)
    {
        if (start == end)
            return false;

        R = (end - start) / incr;

        if(barrier == start)
        {
            double temp = nthreads  * param / 2;
            double b2 = (1 / R) * temp * temp;
            double x = 2 + b2 + sqrt( b2 * (b2 + 4));
            bo_ul F = (R / x) / nthreads;
            bo_ul PF  = F * nthreads;
            bo_ul nbarrier = barrier + (PF * incr);

            if(F <= 1)
            {
                F = 1;
                nbarrier = end;
            }

            bo_ul chunk_tmp =
                __sync_val_compare_and_swap (&ws->chunk_size,
                                             chunk_size,
                                             F);
            if (chunk_tmp != chunk_size) {
                chunk_size = chunk_tmp;
                start = __atomic_load_n (&ws->next, MEMMODEL_RELAXED);
                barrier = __atomic_load_n (&ws->barrier, MEMMODEL_RELAXED);
                continue;
            }
            else {
                chunk_size = F;
            }

            bo_ul barrier_tmp =
                __sync_val_compare_and_swap (&ws->barrier,
                                             barrier,
                                             nbarrier);

            if (barrier_tmp != barrier)
            {
                barrier = barrier_tmp;
                start = __atomic_load_n (&ws->next, MEMMODEL_RELAXED);
                continue;
            }
            else {
                barrier = nbarrier;
            }
        }

        if(__builtin_expect (chunk_size <= R, 1))
            nend = start + chunk_size * incr;
        else
            nend = end;

        long tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
        if (tmp == start)
            break;

        start = tmp;
        barrier = __atomic_load_n (&ws->barrier, MEMMODEL_RELAXED);
        chunk_size = __atomic_load_n (&ws->chunk_size, MEMMODEL_RELAXED);
    }

    *pstart = start;
    *pend = nend;
    return true;
}

inline static bool
bo_iter_ull_fss_next (gomp_ull *pstart, gomp_ull *pend)
{
    struct gomp_thread *thr = gomp_thread ();
    struct gomp_work_share *ws = thr->ts.work_share;
    struct gomp_team *team = thr->ts.team;
    gomp_ull nthreads = team ? team->nthreads : 1;
    gomp_ull start = __atomic_load_n (&ws->next_ull, MEMMODEL_RELAXED);
    gomp_ull end = ws->end_ull;
    gomp_ull incr = ws->incr_ull;
    gomp_ull barrier = __atomic_load_n (&ws->barrier_ull, MEMMODEL_RELAXED);
    gomp_ull chunk_size = __atomic_load_n (&ws->chunk_size_ull, MEMMODEL_RELAXED);
    gomp_ull R = (end - start) / incr;
    gomp_ull nend;

    double param = fss_transform_range(ws->param);

    while (1)
    {
        if (start == end)
            return false;

        if (__builtin_expect (ws->mode, 0) == 0)
            R = (end - start) / incr;
        else
            R = (start - end) / -incr;

        if(barrier == start)
        {
            double temp = nthreads  * param / 2;
            double b2 = (1 / R) * temp * temp;
            double x = 2 + b2 + sqrt( b2 * (b2 + 4));
            gomp_ull F = (R / x) / nthreads;
            gomp_ull PF  = F * nthreads;

            gomp_ull nbarrier;
            if (__builtin_expect (ws->mode, 0) == 0)
                nbarrier = barrier + (PF * incr);
            else
                nbarrier = barrier + (PF * -incr);

            if(F <= 1)
            {
                F = 1;
                nbarrier = end;
            }

            gomp_ull chunk_tmp =
                __sync_val_compare_and_swap (&ws->chunk_size_ull,
                                             chunk_size,
                                             F);
            if (chunk_tmp != chunk_size) {
                chunk_size = chunk_tmp;
                start = __atomic_load_n (&ws->next_ull, MEMMODEL_RELAXED);
                barrier = __atomic_load_n (&ws->barrier_ull, MEMMODEL_RELAXED);
                continue;
            }
            else {
                chunk_size = F;
            }

            gomp_ull barrier_tmp =
                __sync_val_compare_and_swap (&ws->barrier_ull,
                                             barrier,
                                             nbarrier);

            if (barrier_tmp != barrier)
            {
                barrier = barrier_tmp;
                start = __atomic_load_n (&ws->next_ull, MEMMODEL_RELAXED);
                continue;
            }
            else {
                barrier = nbarrier;
            }
        }
        
        if(__builtin_expect (chunk_size <= R, 1))
        {
            if (__builtin_expect (ws->mode, 0) == 0)
                nend = start + chunk_size * incr;
            else
                nend = start + chunk_size * -incr;
        }
        else
            nend = end;

        gomp_ull tmp = __sync_val_compare_and_swap (&ws->next_ull, start, nend);
        if (tmp == start)
            break;

        start = tmp;
        barrier = __atomic_load_n (&ws->barrier_ull, MEMMODEL_RELAXED);
        chunk_size = __atomic_load_n (&ws->chunk_size_ull, MEMMODEL_RELAXED);
    }

    *pstart = start;
    *pend = nend;
    return true;
}

inline static bool
bo_iter_ull_tss_next (gomp_ull *pstart, gomp_ull *pend)
{
    return false;
}

inline static bool
bo_iter_ull_qss_next (gomp_ull *pstart, gomp_ull *pend)
{
    return false;
}

inline static bool
bo_iter_tss_next (long *pstart, long *pend)
{
    return false;
}

inline static bool
bo_iter_qss_next (long *pstart, long *pend)
{
    return false;
}

#endif /* HAVE_SYNC_BUILTINS */
#endif
