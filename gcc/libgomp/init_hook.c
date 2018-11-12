
/* this class' constructor will be called
 * before the linked program's main is executed
 */

#include <stdio.h>
#include <libgomp.h>
#include "bo_scheduling.h"

extern char const* __progname;

void __attribute__ ((constructor))
gomp_init_hook()
{
    struct gomp_task_icv *icv = gomp_icv (false);
    size_t sched_id = (size_t)icv->run_sched_var;

    printf("hook executed!\n");
    printf("current file: %s", __progname);
    bo_load_data(__progname, sched_id);
}

void __attribute__ ((destructor))
gomp_deinit_hook()
{
    bo_save_data(__progname, sched_id);
}
