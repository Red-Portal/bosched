

#ifndef _THIRDPARTY_SCHEDULES_H_
#define _THIRDPARTY_SCHEDULES_H_

#include "libgomp.h"
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <execinfo.h>
#include <sched.h>
#include <pthread.h>

/* workloads. */
extern unsigned __ntasks; /* number of tasks. */
extern unsigned *__tasks;  /* Tasks.           */
extern unsigned __nchunks;

inline static bool
gomp_iter_binlpt_next (long *pstart, long *pend)
{
  int i, j;                   /* Loop index.           */
  int tid;                    /* Thread ID.            */
  int start;                  /* Start of search.      */
  int nthreads;
  struct gomp_thread *thr;    /* Thread.               */
  struct gomp_work_share *ws; /* Work-Share construct. */

  thr		= gomp_thread();
  ws		= thr->ts.work_share;
  nthreads	= ws->nthreads;
  tid		= omp_get_thread_num();
  /* Search for next task. */
  start = __atomic_load_n (&ws->thread_start[tid], MEMMODEL_RELAXED);

  for (i = start; i < __ntasks; i++)
	{
	  if (ws->taskmap[i] == tid)
		{
		  for (j = i + 1; j < __ntasks; j++)
			{
			  if (ws->taskmap[j] != tid)
				  break;
			}

		  long temp = __sync_val_compare_and_swap (&ws->thread_start[tid], start, j);
		  if (temp == start)
			{
			  *pstart = ws->loop_start + i;
			  *pend   = ws->loop_start + j;
			  return true;
			}
		  else
			{
			  start = temp;
			  while(true)
				{
				  long temp = __sync_val_compare_and_swap (&ws->thread_start[tid], start, j);
				  if (temp == start)
					{
					  *pstart = ws->loop_start + start;
					  *pend   = ws->loop_start + j;
					  return true;
					}
				  start = temp;
				  if(start >= j)
					  break;
				}
			}
		}
	}

  long next = __ntasks;
  for(unsigned t = 0; t < nthreads; ++t)
	{
	  start = __atomic_load_n (&ws->thread_start[t], MEMMODEL_RELAXED);
	  next = start < next ? start : next;
	}

  while(true)
	{
	  for(i = next; i < __ntasks; ++i)
		{
		  tid   = ws->taskmap[i];
		  start = __atomic_load_n (&ws->thread_start[tid], MEMMODEL_RELAXED);
		  if(i >= start)
			{
			  long nend = i + 1;
			  long temp = __sync_val_compare_and_swap (&ws->thread_start[tid], start, nend);
			  if (temp == start)
				{
				  *pstart = ws->loop_start + i;
				  *pend   = ws->loop_start + nend;
				  return true;
				}
			  else
				  break;
			}
		}

	  if(i == __ntasks)
		return false;
	}
  return false;
}

inline static bool
gomp_iter_hss_next (long *pstart, long *pend)
{
  unsigned k;                 /* Number of scheduled iterations. */
  long chunksize;             /* Chunksize.                      */
  unsigned chunkweight;       /* Chunk weight.                   */
  struct gomp_thread *thr;    /* Thread.                         */
  struct gomp_team *team;     /* Tead of threads.                */
  struct gomp_work_share *ws; /* Work-Share construct.           */
  int nthreads;               /* Number of threads.              */
  
  /* Get scheduler data.. */
  thr      = gomp_thread();
  team     = thr->ts.team;
  ws       = thr->ts.work_share;
  nthreads = (team != NULL) ? team->nthreads : 1;

  gomp_mutex_lock(&ws->lock);

  /* Comput chunksize. */
  chunksize = ceil(ws->wremaining/(1.5*nthreads));
  if (chunksize < ws->chunk_size)
	chunksize = ws->chunk_size;

  /* Schedule iterations. */
  chunkweight = 0; k = 0;
  for (unsigned i = ws->loop_start; i < __ntasks; i++)
	{
	  unsigned w1;
	  unsigned w2;

	  k++;
	  chunkweight += __tasks[i];

	  w1 = chunkweight;
	  w2 = ((i + 1) < __ntasks) ? chunkweight + __tasks[i + 1] : 0;

	  /* Keep scheduling. */
	  if (w2 <= chunksize)
		continue;

	  /* Best fit. */
	  if (w1 >= chunksize)
		break;

	  /* Best range approximation. */
	  if ((w2 - chunksize) > (chunksize - w1))
		break;
	}

  *pstart = ws->loop_start;
  *pend = ws->loop_start + k;// + 1;

  /* Update scheduler data. */
  ws->loop_start += (k);// + 1);
  ws->wremaining -= chunkweight;

  gomp_mutex_unlock(&ws->lock);

  return ((*pstart == __ntasks) ? false : true);
}

#endif 
