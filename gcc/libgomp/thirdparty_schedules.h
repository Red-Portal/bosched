

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

/**
 * @brief Sets the workload of the next parallel for loop.
 *
 * @param loop_id     The ID of the loop to attach workload information to.
 * @param tasks       Load of iterations.
 * @param ntasks      Number of tasks.
 * @param override    Boolean flag to decide whether we should compute the
 *                    task mapping again or use the preexisting one.
 */

/* inline void omp_set_workload(unsigned loop_id, */
/* 							 unsigned *tasks, */
/* 							 unsigned ntasks, */
/* 							 bool override) */
/* { */
/*   /\* Make sure the loop id is correct.*\/ */
/*   assert((0 <= loop_id) && (loop_id < R_LOOPS)); */

/*   /\* Make sure omp_set_workload() is not called in parallel. *\/ */
/*   assert(gomp_thread()->ts.team_id == 0); */

/*   loops[loop_id].override = override; */
/*   curr_loop = loop_id; */

/*   __tasks = tasks; */
/*   __ntasks = ntasks; */
/* } */

/* #define N 128 */

/* #define exch(a, b)                              \ */
/*   do {                                          \ */
/*     unsigned tmp = (a);                         \ */
/*     (a) = (b);                                  \ */
/*     (b) = tmp;                                  \ */
/*   } while(0); */

/* inline void */
/* insertion(unsigned* map, unsigned* a, unsigned n) */
/* { */
/*   unsigned i, j; /\* Loop indexes.    *\/ */

/*   /\* Sort. *\/ */
/*   for (i = 0; i < (n - 1); i++) */
/* 	{ */
/* 	  for (j = i + 1; j < n; j++) */
/* 		{ */
/* 		  /\* Swap. *\/ */
/* 		  if (a[j] < a[i]) */
/* 			{ */
/* 			  exch(a[i], a[j]); */
/* 			  exch(map[i], map[j]); */
/* 			} */
/* 		} */
/* 	} */
/* } */

/* void quicksort(unsigned *map, unsigned *a, unsigned n) */
/* { */
/*   unsigned i, j; */
/*   unsigned p; */

/*   /\* End recursion. *\/ */
/*   if (n < N) */
/*     { */
/*       insertion(map, a, n); */
/*       return; */
/*     } */

/*   /\* Pivot stuff. *\/ */
/*   p = a[n/2]; */
/*   for (i = 0, j = n - 1; /\* noop *\/ ; i++, j--) */
/*     { */
/*       while (a[i] < p) */
/*         i++; */
/*       while (p < a[j]) */
/*         j--; */
/*       if (i >= j) */
/*         break; */
/*       exch(a[i], a[j]); */
/*       exch(map[i], map[j]); */
/*     } */

/*   quicksort(map, a, i); */
/*   quicksort(map, a + i, n - i); */
/* } */

/* void sort(unsigned *a, */
/* 		  unsigned n, */
/* 		  unsigned *map) */
/* { */
/*   unsigned i; */

/*   /\* Create map. *\/ */
/*   for (i = 0; i < n; i++) */
/*     map[i] = i; */

/*   insertion(map, a, n); */
/* } */

/* inline unsigned* */
/* compute_cummulativesum(unsigned const* a, */
/* 					   unsigned n) */
/* { */
/*   unsigned i; */
/*   unsigned *sum; */

/*   sum = malloc(n*sizeof(unsigned)); */
/*   assert(sum != NULL); */

/*   for (sum[0] = 0, i = 1; i < n; i++) */
/*     sum[i] = sum[i - 1] + a[i - 1]; */

/*   return (sum); */
/* } */

/* inline unsigned* */
/* compute_chunksizes(unsigned const* tasks, */
/* 				   unsigned ntasks, */
/* 				   unsigned nchunks) */
/* { */
/*   unsigned i, k; */
/*   unsigned chunkweight; */
/*   unsigned *chunksizes, *workload; */

/*   chunksizes = calloc(nchunks, sizeof(unsigned)); */
/*   assert(chunksizes != NULL); */

/*   workload = compute_cummulativesum(tasks, ntasks); */

/*   chunkweight = (workload[ntasks - 1] + tasks[ntasks - 1])/nchunks; */

/*   /\* Compute chunksizes. *\/ */
/*   for (k = 0, i = 0; i < ntasks; /\* noop *\/) */
/* 	{ */
/* 	  unsigned j = ntasks; */

/* 	  if (k < (nchunks - 1)) */
/* 		{ */
/* 		  for (j = i + 1; j < ntasks; j++) */
/* 			{ */
/* 			  if (workload[j] - workload[i] > chunkweight) */
/* 				break; */
/* 			} */
/* 		} */

/* 	  chunksizes[k] = j - i; */
/* 	  i = j; */
/* 	  k++; */
/* 	} */

/*   /\* House keeping. *\/ */
/*   free(workload); */

/*   return (chunksizes); */
/* } */

/* /\** */
/*  * @brief Computes chunks. */
/*  *\/ */
/* inline unsigned* */
/* compute_chunks(unsigned const* tasks, */
/* 			   unsigned ntasks, */
/* 			   unsigned const* chunksizes, */
/* 			   unsigned nchunks) */
/* { */
/*   unsigned i, k;    /\* Loop indexes. *\/ */
/*   unsigned *chunks; /\* Chunks.       *\/ */

/*   chunks = calloc(nchunks, sizeof(unsigned)); */
/*   assert(chunks != NULL); */

/*   /\* Compute chunks. *\/ */
/*   for (i = 0, k = 0; i < nchunks; i++) */
/* 	{ */
/* 	  unsigned j; */

/* 	  assert(k <= ntasks); */

/* 	  for (j = 0; j < chunksizes[i]; j++) */
/* 		chunks[i] += tasks[k++]; */
/* 	} */

/*   return (chunks); */
/* } */

/* /\** */
/*  * @brief Bin Packing Longest Processing Time First loop scheduler. */
/*  *\/ */

/* inline unsigned* */
/* binlpt_balance(unsigned* tasks, */
/* 			   unsigned ntasks, */
/* 			   unsigned nthreads) */
/* { */
/*   unsigned i;               /\* Loop index.       *\/ */
/*   unsigned *taskmap;        /\* Task map.         *\/ */
/*   unsigned sortmap[ntasks]; /\* Sorting map.       *\/ */
/*   unsigned *load;           /\* Assigned load.    *\/ */
/*   unsigned *chunksizes;     /\* Chunks sizes.     *\/ */
/*   unsigned *chunks;         /\* Chunks.           *\/ */
/*   unsigned *chunkoff;       /\* Offset to chunks. *\/ */

/*   //printf("[binlpt] Balancing loop %s:%i\n", loops[curr_loop].filename, loops[curr_loop].line); */

/*   /\* Initialize scheduler data. *\/ */
/*   taskmap = calloc(ntasks, sizeof(unsigned)); */
/*   assert(taskmap != NULL); */
/*   load = calloc(nthreads, sizeof(unsigned)); */
/*   assert(load != NULL); */

/*   chunksizes = compute_chunksizes(tasks, ntasks, __nchunks); */
/*   chunks = compute_chunks(tasks, ntasks, chunksizes, __nchunks); */
/*   chunkoff = compute_cummulativesum(chunksizes, __nchunks); */

/*   /\* Sort tasks. *\/ */
/*   sort(chunks, __nchunks, sortmap); */

/*   for (i = __nchunks; i > 0; i--) */
/* 	{ */
/* 	  unsigned j; */
/* 	  unsigned tid; */

/* 	  if (chunks[i - 1] == 0) */
/* 		continue; */

/* 	  tid = 0; */
/* 	  for (j = 1; j < nthreads; j++) */
/* 		{ */
/* 		  if (load[j] < load[tid]) */
/* 			tid = j; */
/* 		} */

/* 	  for (j = 0; j < chunksizes[sortmap[i - 1]]; j++) */
/* 		taskmap[chunkoff[sortmap[i - 1]] + j] = tid; */
/* 	  load[tid] += chunks[i - 1]; */
/* 	} */

/*   /\* House keeping. *\/ */
/*   free(chunkoff); */
/*   free(chunks); */
/*   free(chunksizes); */
/*   free(load); */
/*   return (taskmap); */
/* } */


inline static bool
gomp_iter_binlpt_next (long *pstart, long *pend)
{
  int i, j;                   /* Loop index.           */
  int tid;                    /* Thread ID.            */
  int start;                  /* Start of search.      */
  struct gomp_thread *thr;    /* Thread.               */
  struct gomp_work_share *ws; /* Work-Share construct. */

  thr = gomp_thread();
  ws = thr->ts.work_share;
  tid = omp_get_thread_num();
  /* Search for next task. */
  start = ws->thread_start[tid];
  for (i = start; i < __ntasks; i++)
	{
	  if (ws->taskmap[i] == tid)
		goto found;
	}

  return (false);

 found:

  for (j = i + 1; j < __ntasks; j++)
	{
	  if (ws->taskmap[j] != tid)
		break;
	}

  ws->thread_start[tid] = j;
  *pstart = ws->loop_start + i;
  *pend = ws->loop_start + j;

  return (true);
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
  thr = gomp_thread();
  team = thr->ts.team;
  ws = thr->ts.work_share;
  nthreads = (team != NULL) ? team->nthreads : 1;

  //#ifndef HAVE_SYNC_BUILTINS
  gomp_mutex_lock(&ws->lock);
  //#endif

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
  *pend = ws->loop_start + k + 1;

  /* Update scheduler data. */
  ws->loop_start += k;
  ws->wremaining -= chunkweight;

  //#ifndef HAVE_SYNC_BUILTINS
  gomp_mutex_unlock(&ws->lock);
  //#endif

  return ((*pstart == __ntasks) ? false : true);
}

#endif 
