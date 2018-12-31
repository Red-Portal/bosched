
/* this class' constructor will be called
 * before the linked program's main is executed
 */

#include <stdio.h>
#include <libgomp.h>
#include "bo_scheduling.h"

extern char const* __progname;
static void* _context;

void __attribute__ ((constructor))
gomp_init_hook()
{
    struct gomp_task_icv *icv = gomp_icv (false);
    size_t sched_id = (size_t)icv->run_sched_var;

    printf("init hook executed!\n");
    printf("current file: %s\n", __progname);
    _context = bo_load_data(__progname, sched_id);
}

void __attribute__ ((destructor))
gomp_deinit_hook()
{
    printf("deinit hook executed!\n");
    struct gomp_task_icv *icv = gomp_icv (false);
    size_t sched_id = (size_t)icv->run_sched_var;
    bo_save_data(_context, __progname, sched_id);
}
