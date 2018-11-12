
#ifndef BO_SCHEDULING_H
#define BO_SCHEDULING_H 1

#include "libgomp.h"

extern void bo_load_data(char const* progname,
                         size_t sched_id);

extern void bo_save_data(char const* progname,
                         size_t sched_id);

extern double bo_schedule_parameter(unsigned long long region_id);

extern void bo_schedule_begin(unsigned long long region_id);

extern void bo_schedule_end(unsigned long long region_id);

inline static bool is_bo_schedule(enum gomp_schedule_type sched)
{
    return sched == BO_FSS || sched == BO_CSS || sched == BO_QSS;
}

inline static bool is_parameterized(enum gomp_schedule_type sched)
{
    return is_bo_schedule(sched) || sched == FS_CSS;
}


#endif
