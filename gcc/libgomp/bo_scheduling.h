
#ifndef BO_SCHEDULING_H
#define BO_SCHEDULING_H 1

#include <string.h>
#include "libgomp.h"

extern void bo_workload_profile_init(unsigned long long num_tasks);

extern void bo_workload_profile_start(long iteration);

extern void bo_workload_profile_stop();

extern void bo_record_iteration_start();

extern void bo_record_iteration_stop();

extern void bo_load_data(char const* progname,
						 size_t sched_id);

extern void bo_binlpt_load_loop(unsigned long long region_id,
								int* out_override,
								unsigned* inout_task_map);

extern void bo_save_data(char const* progname,
						 size_t sched_id);

extern double bo_schedule_parameter(unsigned long long region_id,
									int is_bo);

extern void bo_schedule_begin(unsigned long long region_id,
							  unsigned long long N,
							  long num_tasks );

extern void bo_schedule_end(unsigned long long region_id);

inline static bool is_bo_schedule(enum gomp_schedule_type sched)
{
  return sched == BO_FSS
	|| sched == BO_CSS
	|| sched == BO_TAPE
	|| sched == BO_TSS;
}

inline static bool is_parameterized(enum gomp_schedule_type sched)
{
  return is_bo_schedule(sched)
	|| sched == FS_CSS
	|| sched == FS_FSS
	|| sched == FS_TAPE;
}

#endif
