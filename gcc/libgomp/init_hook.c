
/* this class' constructor will be called
 * before the linked program's main is executed
 */

#include <stdio.h>

extern char const* __progname;

void __attribute__ ((constructor))
gomp_init_hook()
{
    printf("hook executed!\n");
    printf("current file: %s", __progname);
}



void __attribute__ ((destructor))
gomp_deinit_hook()
{

}

