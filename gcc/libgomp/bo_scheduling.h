
#ifndef __BO_SCHED__
#define __BO_SCHED__

#include "libgomp.h"

extern double bo_schedule_parameter(unsigned long long region_id);

// newly added schedules 
// FS_AF,
// FS_FAC2,
// FS_FSS,
// FS_TSS,
// FS_CSS,
// FS_QSS,
// BO_FSS,
// BO_CSS,
// BO_QSS,

static inline bool is_bo_schedule(gomp_schedule_type sched)
{
    return sched == BO_FSS || sched == BO_CSS || sched == BO_QSS;
}

#endif
