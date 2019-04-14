/* Copyright (C) 2005-2014 Free Software Foundation, Inc.
   Contributed by Richard Henderson <rth@redhat.com>.

   This file is part of the GNU OpenMP Library (libgomp).

   Libgomp is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.

   Libgomp is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
   FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
   more details.

   Under Section 7 of GPL version 3, you are granted additional
   permissions described in the GCC Runtime Library Exception, version
   3.1, as published by the Free Software Foundation.

   You should have received a copy of the GNU General Public License and
   a copy of the GCC Runtime Library Exception along with this program;
   see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
   <http://www.gnu.org/licenses/>.  */

/* This file handles the LOOP (FOR/DO) construct.  */

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "libgomp.h"
#include <math.h>
#include <time.h>
//#define EXPORT __attribute__((visibility("default")))


	
/*============================================================================*
 * Workload Information                                                       *
 *============================================================================*/

/**
 * @brief Tasks.
 */
unsigned *__tasks;

/**
 * @brief Number of tasks.
 */
unsigned __ntasks;


// MY VARIABLES LOL
//extern double *timearray;
/**
 * @brief Sets the workload of the next parallel for loop.
 *
 * @param tasks  Load of iterations.
 * @param ntasks Number of tasks.
 */

double omp_set_chunksize_fsc(double sigma, double h, int n, int p)
{
	return pow(((sqrt(2)*n*h)/(sigma*p*sqrt(log(p)))),(2/3));
}
//ialias (omp_set_chunksize_fsc)
void omp_set_workload(unsigned *tasks, unsigned ntasks)
{
	__tasks = tasks;
	__ntasks = ntasks;
} 

unsigned __nchunks;

/*============================================================================*
 * Workload Sorting                                                           *
 *============================================================================*/

#define N 128

/*
 * Exchange two numbers.
 */
#define exch(a, b, t) \
	{ (t) = (a); (a) = (b); (b) = (t); }

/*
 * Insertion sort.
 */
static void insertion(unsigned *map, unsigned *a, unsigned n)
{
	unsigned t;    /* Temporary value. */
	unsigned i, j; /* Loop indexes.    */
	
	/* Sort. */
	for (i = 0; i < (n - 1); i++)
	{
		for (j = i + 1; j < n; j++)
		{
			/* Swap. */
			if (a[j] < a[i])
			{
				exch(a[i], a[j], t);
				exch(map[i], map[j], t);
			}
		}
	}
}

/*
 * Quicksort algorithm.
 */
void quicksort(unsigned *map, unsigned *a, unsigned n)
{
	unsigned i, j;
	unsigned p, t;
    
    /* End recursion. */
	if (n < N)
	{
		insertion(map, a, n);
		return;
	}
    
	/* Pivot stuff. */
	p = a[n/2];
	for (i = 0, j = n - 1; /* noop */ ; i++, j--)
	{
		while (a[i] < p)
			i++;
		while (p < a[j])
			j--;
		if (i >= j)
			break;
		exch(a[i], a[j], t);
		exch(map[i], map[j], t);
	}
    
	quicksort(map, a, i);
	quicksort(map, a + i, n - i);
}
 
/*
 * Sorts an array of numbers.
 */
void sort(unsigned *a, unsigned n, unsigned *map)
{
	unsigned i;	

	/* Create map. */
	for (i = 0; i < n; i++)
		map[i] = i;

	insertion(map, a, n);
} 

/*============================================================================*
 * SRR Loop Scheduler                                                         *
 *============================================================================*/

/**
 * @brief Smart Round-Robin loop scheduler.
 *
 * @param tasks    Target tasks.
 * @param ntasks   Number of tasks.
 * @param nthreads Number of threads.
 *
 * @returns Iteration scheduling map.
 */
static unsigned *srr_balance(unsigned *tasks, unsigned ntasks, unsigned nthreads)
{
	unsigned k;               /* Scheduling offset. */
	unsigned tid;             /* Current thread ID. */
	unsigned i, j;            /* Loop indexes.      */
	unsigned *taskmap;        /* Task map.          */
	unsigned sortmap[ntasks]; /* Sorting map.       */
	unsigned load[ntasks];    /* Assigned load.     */

	/* Initialize scheduler data. */
	taskmap = malloc(ntasks*sizeof(unsigned));
	assert(taskmap != NULL);
	memset(load, 0, ntasks * sizeof(unsigned));

	/* Sort tasks. */
	sort(tasks, ntasks, sortmap);

	/* Assign tasks to threads. */
	tid = 0;
	k = ntasks & 1;

	for (i = k; i < k + (ntasks - k); i++)
	{
		unsigned l = sortmap[i];
		unsigned r = sortmap[ntasks - ((i - k) + 1)];

		taskmap[l] = tid;
		taskmap[r] = tid;

		load[tid] += tasks[l] + tasks[r];

		/* Wrap around. */
		tid = (tid + 1)%nthreads;
	}

	/* Assign remaining tasks. */
	for (i = k; i > 0; i--)
	{
		unsigned leastoverload;

		/* Find least overload thread. */
		leastoverload = 0;
		for (j = 1; j < nthreads; j++)
		{
			if (load[j] < load[leastoverload])
				leastoverload = j;
		}

		taskmap[sortmap[i - 1]] = leastoverload;

		load[leastoverload] += tasks[sortmap[i - 1]];
	}

	return (taskmap);
}

/*============================================================================*
 * BIN+LPT Loop Scheduler                                                     *
 *============================================================================*/

/**
 * @brief Computes the cummulative sum of an array.
 *
 * @param a Target array.
 * @param n Size of target array.
 *
 * @returns Commulative sum.
 */
static unsigned *compute_cummulativesum(const unsigned *a, unsigned n)
{
	unsigned i;
	unsigned *sum;

	sum = malloc(n*sizeof(unsigned));
	assert(sum != NULL);

	for (sum[0] = 0, i = 1; i < n; i++)
		sum[i] = sum[i - 1] + a[i - 1];

	return (sum);
}

/**
 * @brief Computes chunk sizes.
 *
 * @param tasks   Target tasks.
 * @param ntasks  Number of tasks.
 * @param nchunks Number of chunks.
 *
 * @returns Chunk sizes.
 */
static unsigned *compute_chunksizes(const unsigned *tasks, unsigned ntasks, unsigned nchunks)
{
	unsigned i, k;
	unsigned chunkweight;
	unsigned *chunksizes, *workload;

	chunksizes = calloc(nchunks, sizeof(unsigned));
	assert(chunksizes != NULL);
	
	workload = compute_cummulativesum(tasks, ntasks);

	chunkweight = (workload[ntasks - 1] + tasks[ntasks - 1])/nchunks;

	/* Compute chunksizes. */
	for (k = 0, i = 0; i < ntasks; /* noop */)
	{
		unsigned j = ntasks;

		if (k < (nchunks - 1))
		{
			for (j = i + 1; j < ntasks; j++)
			{
				if (workload[j] - workload[i] > chunkweight)
					break;
			}
		}

		chunksizes[k] = j - i;
		i = j;
		k++;
	}

	/* House keeping. */
	free(workload);

	return (chunksizes);
}

/**
 * @brief Computes chunks.
 */
static unsigned *compute_chunks(const unsigned *tasks, unsigned ntasks, const unsigned *chunksizes, unsigned nchunks)
{
	unsigned i, k;    /* Loop indexes. */
	unsigned *chunks; /* Chunks.       */

	chunks = calloc(nchunks, sizeof(unsigned));
	assert(chunks != NULL);

	/* Compute chunks. */
	for (i = 0, k = 0; i < nchunks; i++)
	{
		unsigned j;

		assert(k <= ntasks);

		for (j = 0; j < chunksizes[i]; j++)
			chunks[i] += tasks[k++];
	}

	return (chunks);
}

/**
 * @brief Bin Packing Longest Processing Time First loop scheduler.
 */
static unsigned *binlpt_balance(unsigned *tasks, unsigned ntasks, unsigned nthreads)
{
	unsigned i;               /* Loop index.       */
	unsigned *taskmap;        /* Task map.         */
	unsigned sortmap[ntasks]; /* Sorting map.       */
	unsigned *load;           /* Assigned load.    */
	unsigned *chunksizes;     /* Chunks sizes.     */
	unsigned *chunks;         /* Chunks.           */
	unsigned *chunkoff;       /* Offset to chunks. */

	/* Initialize scheduler data. */
	taskmap = calloc(ntasks, sizeof(unsigned));
	assert(taskmap != NULL);
	load = calloc(nthreads, sizeof(unsigned));
	assert(load != NULL);

	chunksizes = compute_chunksizes(tasks, ntasks, __nchunks);
	chunks = compute_chunks(tasks, ntasks, chunksizes, __nchunks);
	chunkoff = compute_cummulativesum(chunksizes, __nchunks);

	/* Sort tasks. */
	sort(chunks, __nchunks, sortmap);
	
	for (i = __nchunks; i > 0; i--)
	{
		unsigned j;
		unsigned tid;

		if (chunks[i - 1] == 0)
			continue;

		tid = 0;
		for (j = 1; j < nthreads; j++)
		{
			if (load[j] < load[tid])
				tid = j;
		}

		for (j = 0; j < chunksizes[sortmap[i - 1]]; j++)
			taskmap[chunkoff[sortmap[i - 1]] + j] = tid;
		load[tid] += chunks[i - 1];
	}

	/* House keeping. */
	free(chunkoff);
	free(chunks);
	free(chunksizes);
	free(load);
	
	return (taskmap);
}

/*============================================================================*
 * Hacked LibGomp Routines                                                    *
 *============================================================================*/

/* Initialize the given work share construct from the given arguments.  */

static inline void
gomp_loop_init (struct gomp_work_share *ws, long start, long end, long incr,
		enum gomp_schedule_type sched, long chunk_size, unsigned num_threads)
{
  ws->sched = sched;
  ws->chunk_size = chunk_size;
  /* Canonicalize loops that have zero iterations to ->next == ->end.  */
  ws->end = ((incr > 0 && start > end) || (incr < 0 && start < end))
	    ? start : end;
  ws->incr = incr;
  ws->next = start;


  switch (sched) {
  case GFS_DYNAMIC:
    ws->chunk_size *= incr;

#ifdef HAVE_SYNC_BUILTINS
    {
      /* For dynamic scheduling prepare things to make each iteration
	 faster.  */
      struct gomp_thread *thr = gomp_thread ();
      struct gomp_team *team = thr->ts.team;
      long nthreads = team ? team->nthreads : 1;
  	//  clock_gettime(CLOCK_MONOTONIC, &ws->dyntime);
  	 // ws->maxiters = end-start;
 // ws->dynarray = (double*)malloc(sizeof(double) * (ws->maxiters+5));

//	ws->globalfactcounter = -1;
      if (__builtin_expect (incr > 0, 1))
	{
	  /* Cheap overflow protection.  */
	  if (__builtin_expect ((nthreads | ws->chunk_size)
				>= 1UL << (sizeof (long)
					   * __CHAR_BIT__ / 2 - 1), 0))
	    ws->mode = 0;
	  else
	    ws->mode = ws->end < (LONG_MAX
				  - (nthreads + 1) * ws->chunk_size);
	}
      /* Cheap overflow protection.  */
      else if (__builtin_expect ((nthreads | -ws->chunk_size)
				 >= 1UL << (sizeof (long)
					    * __CHAR_BIT__ / 2 - 1), 0))
	ws->mode = 0;
      else
	ws->mode = ws->end > (nthreads + 1) * -ws->chunk_size - LONG_MAX;
    }
#endif
    break;


  case GFS_RAND:
  {
    if (num_threads == 0)
	  {
		  struct gomp_thread *thr = gomp_thread ();
		  struct gomp_team *team = thr->ts.team;
		  num_threads = (team != NULL) ? team->nthreads : 1;
	  }
  printf("rand");
  ws->randmin = (end-start)/(num_threads*100);
  ws->randmax = (end-start)/(num_threads*2);

if(getenv("RANDMIN"))  ws->randmin = strtod(getenv("RANDMIN"),NULL);
if(getenv("RANDMAX"))  ws->randmax = strtod(getenv("RANDMAX"),NULL);
 if(ws->randmin < 1) ws->randmin = 1;
 if(ws->randmax < ws->randmin) ws->randmax = ws->randmin + 1; 
break;
  }
  case GFS_AWF:
  {
  printf("awf");
  break;
  }
  case GFS_FACT:
   {
//printf("fac");
    if (num_threads == 0)
	  {
		  struct gomp_thread *thr = gomp_thread ();
		  struct gomp_team *team = thr->ts.team;
		  num_threads = (team != NULL) ? team->nthreads : 1;
	  }

	ws->globalfactcounter = -1;
	ws->maxworkload = 0;
	 ws->facsigma = strtod(getenv("SIGMA"),NULL);
	 ws->facmue = strtod(getenv("MUE"),NULL);
	 
   ws->maxworkload = (end-start); 
   break;
}
  case GFS_FAC2:
   {
//printf("fac2");
    if (num_threads == 0)
	  {
		  struct gomp_thread *thr = gomp_thread ();
		  struct gomp_team *team = thr->ts.team;
		  num_threads = (team != NULL) ? team->nthreads : 1;
	  }

	ws->globalfactcounter = -1;
	ws->maxworkload = 0;

   ws->maxworkload = (end-start);
   
   break;
}
  case GFS_TAPE:
   {//printf("tape");

     if (num_threads == 0)
	  {
		  struct gomp_thread *thr = gomp_thread ();
		  struct gomp_team *team = thr->ts.team;
		  num_threads = (team != NULL) ? team->nthreads : 1;
	  }
	 double sigma = 0;
	 double mue = 0;
         double alpha = 1.3;
	 sigma = strtod(getenv("SIGMA"),NULL);
	 alpha = strtod(getenv("ALPHA"),NULL);
	 mue = strtod(getenv("MUE"),NULL);
	 if (mue == 0) mue = 0.0000000001;
	 ws->va = (alpha*sigma)/mue;
      break;
}
  case GFS_FRAC:
   {
      break;
}
  case GFS_WFAC:
   {
   //printf("wfac");
     if (num_threads == 0)
	  {
		  struct gomp_thread *thr = gomp_thread ();
		  struct gomp_team *team = thr->ts.team;
		  num_threads = (team != NULL) ? team->nthreads : 1;
	  }
	ws->globalWFACcounter = -1;
	ws->WFACworkload = 0;
	//printf("numthreads: %i, thrnum: %i\n, wfacinit: %i\n\n", num_threads,omp_get_thread_num(), ws->WFACinit);	
	//printf("wfaci: %i\n\n", ws->WFACinit);
	//if (ws->WFACinit > 0){ printf("init done");} 
	//else printf("init not done\n\n");
	//printf("dorde2\n\n");
	//if (ws->WFACinit == 0){
	//printf("is it crashing here?");
	ws->WFACarray = (double*)malloc(sizeof(double) * num_threads); 
	//__sync_add_and_fetch(&ws->WFACinit,1);
	//}
	
	//printf("it worked\n\n");
	for (int i= 0; i < num_threads; i++) ws->WFACarray[i] = 1;
	//char * pch;
	char* env_weights = getenv("WEIGHTS");
	for (int i = 0; i < num_threads; i++) {
		ws->WFACarray[i] = strtod(env_weights, &env_weights);
		}
	//char * input;
	//input = (char*)malloc(sizeof(char)*num_threads);
	//input = getenv("WEIGHTS");
	//pch = strtok (getenv("WEIGHTS")," ");
	//int tempvarwfac = 0;
	//while(tempvarwfac < num_threads) {
		//asdf[tempvarwfac] = strtod(pch,NULL);
	//	asdf[tempvarwfac] = strtod(input[tempvarwfac], NULL); 
	//	tempvarwfac++;
		//pch = strtok(NULL, " ");
		//printf("yo!\n");
	//}
	//for (int eee = 0; eee < 4; eee++) {
	//printf("//eee: %d\n",eee);
	//printf("000multip: %f\n",ws->WFACarray[eee]);}
		
	ws->WFACworkload = (end-start);
	
      break;
}
  case GFS_BOLD:
   {//printf("bold");
   if (num_threads == 0)
	  {
		  struct gomp_thread *thr = gomp_thread ();
		  struct gomp_team *team = thr->ts.team;
		  num_threads = (team != NULL) ? team->nthreads : 1;
	  }
  double boldsigma = 0;
  ws->totalspeed = 0;
  boldsigma = strtod(getenv("SIGMA"),NULL);
  ws->boldmue = strtod(getenv("MUE"),NULL);
  ws->boldh = strtod(getenv("BOLDH"),NULL);
  //printf("numthreads1: %i", num_threads);
  if(ws->boldmue == 0) ws->boldmue = 0.0000000001;
  
  //incr = ws->incr;
  ws->boldtset=false;
  ws->bolda = 2*((boldsigma/ws->boldmue)*(boldsigma/ws->boldmue));
  ws->boldb = 8*ws->bolda*log(8*ws->bolda);
  if (ws->boldb > 0) ws->ln_b = log(ws->boldb);
  ws->p_inv = 1.0/num_threads;
  ws->c1 = ws->boldh/(ws->boldmue*log(2));
  ws->c2 = sqrt(2*M_PI)*ws->c1;
  ws->c3 = log(ws->c2);
  ws->boldm = end;
  ws->boldn = end;
  //if (!ws->boldinit){
  //printf("hi\n");
  ws->boldarray = (double*)malloc(sizeof(double) * num_threads);
  ws->speedarray = (double*)malloc(sizeof(double) * num_threads);
  ws->timearray = (struct timespec*)malloc(sizeof(struct timespec) * num_threads);
  //printf("everything initialized\n");
  //}
  struct timespec timerhelper;
  clock_gettime(CLOCK_MONOTONIC, &timerhelper);
  clock_gettime(CLOCK_MONOTONIC, &ws->boldtime);
  //double timerhelper2 = timerhelper.tv_sec + (double)(timerhelper.tv_nsec)*0.000000001;
//printf("timer: %f, numthreads: %i\n", timerhelper, num_threads);
  for (int i = 0; i < num_threads; i++){ ws->boldarray[i] = 0;}
  for (int i = 0; i < num_threads; i++){ ws->speedarray[i] = 0;}
  for (int i = 0; i < num_threads; i++){ ws->timearray[i] = timerhelper;} 
  //printf("the arrays should work now\n");
  //§printf("boldstuff: %f,%f,%f,%f\n", boldarray[1], speedarray[1], timearray[1], timerhelper);
  //ws->boldinit = true;
      break;
}
   case GFS_TRAP:
   {//printf("trap");
	if (num_threads == 0)
	  {
		  struct gomp_thread *thr = gomp_thread ();
		  struct gomp_team *team = thr->ts.team;
		  num_threads = (team != NULL) ? team->nthreads : 1;
	  }

	ws->trapcounter = -1;
	ws->decr_delta = 0;
	ws->startsize = (end-start)/(num_threads*2);
	int endsize =  (end-start)/(num_threads*100);
	if(endsize < 1) endsize = 1;	
	 // printf("Trapezoid started \n");
	  start = ws->next;
	  end = ws->end;
	  //incr = ws->incr;
	  chunk_size = ws->chunk_size;
	  if(getenv("TRAPSTART")) ws->startsize = atoi(getenv("TRAPSTART"));
	  if(getenv("TRAPEND")) endsize = atoi(getenv("TRAPEND"));
	  //testn = (end - start) / incr;
	  int big_n = (2*end)/(ws->startsize+endsize);
	 //printf("hello there from thread nr: %i\n",omp_get_thread_num()); 
	  if (big_n != 1){ ws->decr_delta = (double) ((double) ws->startsize - (double) endsize)/( (double) big_n - 1.0);}
	  else ws->decr_delta = 1;
	  //printf( "ddd%f, %i\n",decr_delta, omp_get_thread_num());
	  //decr_delta = (startsize - endsize)/(big_n - 1);
//	  printf("start: %i, end: %i, numthr: %i\n",startsize,endsize, num_threads);
   break;
	}
  case GFS_FSC:
  {//printf("fsc");
	if (num_threads == 0)
	  {
		  struct gomp_thread *thr = gomp_thread ();
		  struct gomp_team *team = thr->ts.team;
		  num_threads = (team != NULL) ? team->nthreads : 1;
	  }

	double h = strtod(getenv("FSCH"),NULL);
	double sigma = strtod(getenv("SIGMA"),NULL);
	if (sigma == 0) sigma = 0.0000000001;
	//if (num_threads == 0) num_threads = 2;
	double test = (sqrt(2.0)*(double)end*h)/(sigma*(double)num_threads*sqrt(log((double)num_threads))); 
  	ws->fscsize = pow(test,(2.0/3.0));
	if (ws->fscsize > end) ws->fscsize = end;
	if (ws->fscsize < 1) ws->fscsize = 1;
//	printf("end: %ld, h: %f, sigma:%f, nthr: %i, fscsize: %i, test: %f\n", end, h, sigma, num_threads, fscsize,test);
  break;
  }
  case GFS_AFAC:
  {//printf("afac");
  if (num_threads == 0)
	  {
		  struct gomp_thread *thr = gomp_thread ();
		  struct gomp_team *team = thr->ts.team;
		  num_threads = (team != NULL) ? team->nthreads : 1;
	  }

  ws->afacounter = -1;
  //if (!boldinit){
  //printf("hi\n");
  
  end = ws->end;
  ws->afacarray = (double*)malloc(sizeof(double) * num_threads);
  ws->afacspeed = (double*)malloc(sizeof(double) * num_threads * end);
  ws->afacmu = (double*)malloc(sizeof(double) * num_threads);
   
  ws->afacsigma = (double*)malloc(sizeof(double) * num_threads);
  ws->afactimetaken = (long*)malloc(sizeof(long) * num_threads);
  ws->afacitercount = (int*)malloc(sizeof(int) * num_threads);
  //timearray = (double*)malloc(sizeof(double) * num_threads);
  //printf("everything initialized\n");
  //}
  //double timerhelper = time(NULL);
  //printf("timer: %f, numthreads: %i\n", timerhelper, num_threads);
  for (int i = 0; i < num_threads; i++){ ws->afacarray[i] = 0;}
  for (int i = 0; i < num_threads; i++){ ws->afacmu[i] = 1;}
  for (int i = 0; i < num_threads; i++){ ws->afactimetaken[i] = 0;}
  for (int i = 0; i < num_threads*end; i++){ ws->afacspeed[i] = 0;}
  //for (int i = 0; i < num_threads; i++){ timearray[i] = timerhelper;} 
  for (int i = 0; i < num_threads; i++){ ws->afacsigma[i] = 0;}
  for (int i = 0; i < num_threads; i++){ ws->afacitercount[i] = 0;}
	//double h = strtod(getenv("BOLDH"),NULL);
	//double sigma = strtod(getenv("SIGMA"),NULL);
  	//fscsize = pow(((sqrt(2)*end*h)/(sigma*num_threads*sqrt(log(num_threads)))),(2/3));
  break;
  }
  case GFS_BINLPT:
	{
	  __nchunks = chunk_size;
	  if (__nchunks == 1)
		  __nchunks = __ntasks;
	}
  case GFS_SRR:
    {
      unsigned *(*balance)(unsigned *, unsigned, unsigned);
      balance = (sched == GFS_SRR) ? srr_balance : binlpt_balance;
      if (num_threads == 0)
	  {
		  struct gomp_thread *thr = gomp_thread ();
		  struct gomp_team *team = thr->ts.team;
		  num_threads = (team != NULL) ? team->nthreads : 1;
	  }

      ws->taskmap = balance(__tasks, __ntasks, num_threads);
      ws->loop_start = start;
      ws->thread_start = (unsigned *) calloc(num_threads, sizeof(int));
    }
    break;

  default:
    break;
  }
}

static bool
gomp_loop_static_start (long start, long end, long incr, long chunk_size,
			long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();

  thr->ts.static_trip = 0;
  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_STATIC, chunk_size, 0);
      gomp_work_share_init_done ();
    }

  return !gomp_iter_static_next (istart, iend);
}

static bool
gomp_loop_dynamic_start (long start, long end, long incr, long chunk_size,
			 long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_DYNAMIC, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_dynamic_next (istart, iend);
#else
  gomp_mutex_lock (&thr->ts.work_share->lock);
  ret = gomp_iter_dynamic_next_locked (istart, iend);
  gomp_mutex_unlock (&thr->ts.work_share->lock);
#endif

  return ret;
}

static bool
gomp_loop_guided_start (long start, long end, long incr, long chunk_size,
			long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_GUIDED, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_guided_next (istart, iend);
#else
  gomp_mutex_lock (&thr->ts.work_share->lock);
  ret = gomp_iter_guided_next_locked (istart, iend);
  gomp_mutex_unlock (&thr->ts.work_share->lock);
#endif

  return ret;
}
static bool
gomp_loop_trap_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{

  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_TRAP, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_trap_next (istart, iend);
#endif

  return ret;
}
static bool
gomp_loop_rand_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{

  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_RAND, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_rand_next (istart, iend);
#endif

  return ret;
}
static bool
gomp_loop_fac2_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{

  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_FAC2, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_fac2_next (istart, iend);
#endif

  return ret;
}
static bool
gomp_loop_fact_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{

  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_FACT, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_fact_next (istart, iend);
#endif

  return ret;
}
static bool
gomp_loop_tape_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{

  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_TAPE, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_tape_next (istart, iend);
#endif

  return ret;
}
static bool
gomp_loop_frac_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{

  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_FRAC, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_frac_next (istart, iend);
#endif

  return ret;
}
static bool
gomp_loop_wfac_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{

  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_WFAC, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_wfac_next (istart, iend);
#endif

  return ret;
}

static bool
gomp_loop_bold_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{

  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_BOLD, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_bold_next (istart, iend);
#endif

  return ret;
}
static bool
gomp_loop_fsc_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{

  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_FSC, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_fsc_next (istart, iend);
#endif

  return ret;
}
static bool
gomp_loop_afac_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{

  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_AFAC, chunk_size, 0);
      gomp_work_share_init_done ();
    }

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_afac_next (istart, iend);
#endif

  return ret;
}
static bool
gomp_loop_binlpt_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_BINLPT, chunk_size, 0);
      gomp_work_share_init_done ();
    }

  ret = gomp_iter_binlpt_next (istart, iend);

  return ret;
}

static bool
gomp_loop_srr_start (long start, long end, long incr, long chunk_size,
		       long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (false))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_SRR, chunk_size, 0);
      gomp_work_share_init_done ();
    }

  ret = gomp_iter_srr_next (istart, iend);

  return ret;
}

bool
GOMP_loop_runtime_start (long start, long end, long incr,
			 long *istart, long *iend)
{
  struct gomp_task_icv *icv = gomp_icv (false);
  struct gomp_thread *thr = gomp_thread ();
 struct gomp_team *team = thr->ts.team;
 int num_threads = (team != NULL) ? team->nthreads : 1;
  if(end <= 2*num_threads && icv->run_sched_var != GFS_STATIC) 
  {
  //printf("special case\n");
  return gomp_loop_static_start (start, end, incr, icv->run_sched_modifier, istart, iend);
  }
  
switch (icv->run_sched_var)
    {
    case GFS_STATIC:
      return gomp_loop_static_start (start, end, incr, icv->run_sched_modifier,
				     istart, iend);
    case GFS_DYNAMIC:
      return gomp_loop_dynamic_start (start, end, incr, icv->run_sched_modifier,
				      istart, iend);
    case GFS_GUIDED:
      return gomp_loop_guided_start (start, end, incr, icv->run_sched_modifier,
				     istart, iend);
    case GFS_FACT:
      return gomp_loop_fact_start (start, end, incr, icv->run_sched_modifier, istart, iend); 
    case GFS_TAPE:
      return gomp_loop_tape_start (start, end, incr, icv->run_sched_modifier, istart, iend); 
    case GFS_FRAC:
      return gomp_loop_frac_start (start, end, incr, icv->run_sched_modifier, istart, iend); 
    case GFS_WFAC:
      return gomp_loop_wfac_start (start, end, incr, icv->run_sched_modifier, istart, iend); 
    case GFS_BOLD:
      return gomp_loop_bold_start (start, end, incr, icv->run_sched_modifier, istart, iend); 
    case GFS_TRAP:
      return gomp_loop_trap_start (start, end, incr, icv->run_sched_modifier, istart, iend); 
    case GFS_FSC:
      return gomp_loop_fsc_start (start, end, incr, icv->run_sched_modifier, istart, iend);
    case GFS_AFAC:
      return gomp_loop_afac_start (start, end, incr, icv->run_sched_modifier, istart, iend);
    case GFS_BINLPT:
      return gomp_loop_binlpt_start (start, end, incr, icv->run_sched_modifier, istart, iend);
    case GFS_SRR:
      return gomp_loop_srr_start (start, end, incr, icv->run_sched_modifier, istart, iend);
    case GFS_FAC2:
      return gomp_loop_fac2_start (start, end, incr, icv->run_sched_modifier, istart, iend);
    case GFS_RAND:
      return gomp_loop_rand_start (start, end, incr, icv->run_sched_modifier, istart, iend);
    case GFS_AUTO:
      /* For now map to schedule(static), later on we could play with feedback
	 driven choice.  */
      return gomp_loop_static_start (start, end, incr, 0, istart, iend);
    default:
      abort ();
    }
}

/* The *_ordered_*_start routines are similar.  The only difference is that
   this work-share construct is initialized to expect an ORDERED section.  */

static bool
gomp_loop_ordered_static_start (long start, long end, long incr,
				long chunk_size, long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();

  thr->ts.static_trip = 0;
  if (gomp_work_share_start (true))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_STATIC, chunk_size, 0);
      gomp_ordered_static_init ();
      gomp_work_share_init_done ();
    }

  return !gomp_iter_static_next (istart, iend);
}

static bool
gomp_loop_ordered_dynamic_start (long start, long end, long incr,
				 long chunk_size, long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (true))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_DYNAMIC, chunk_size, 0);
      gomp_mutex_lock (&thr->ts.work_share->lock);
      gomp_work_share_init_done ();
    }
  else
    gomp_mutex_lock (&thr->ts.work_share->lock);

  ret = gomp_iter_dynamic_next_locked (istart, iend);
  if (ret)
    gomp_ordered_first ();
  gomp_mutex_unlock (&thr->ts.work_share->lock);

  return ret;
}

static bool
gomp_loop_ordered_guided_start (long start, long end, long incr,
				long chunk_size, long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  if (gomp_work_share_start (true))
    {
      gomp_loop_init (thr->ts.work_share, start, end, incr,
		      GFS_GUIDED, chunk_size, 0);
      gomp_mutex_lock (&thr->ts.work_share->lock);
      gomp_work_share_init_done ();
    }
  else
    gomp_mutex_lock (&thr->ts.work_share->lock);

  ret = gomp_iter_guided_next_locked (istart, iend);
  if (ret)
    gomp_ordered_first ();
  gomp_mutex_unlock (&thr->ts.work_share->lock);

  return ret;
}

bool
GOMP_loop_ordered_runtime_start (long start, long end, long incr,
				 long *istart, long *iend)
{
  struct gomp_task_icv *icv = gomp_icv (false);
  switch (icv->run_sched_var)
    {
    case GFS_STATIC:
      return gomp_loop_ordered_static_start (start, end, incr,
					     icv->run_sched_modifier,
					     istart, iend);
    case GFS_DYNAMIC:
      return gomp_loop_ordered_dynamic_start (start, end, incr,
					      icv->run_sched_modifier,
					      istart, iend);
    case GFS_GUIDED:
      return gomp_loop_ordered_guided_start (start, end, incr,
					     icv->run_sched_modifier,
					     istart, iend);
    case GFS_AUTO:
      /* For now map to schedule(static), later on we could play with feedback
	 driven choice.  */
      return gomp_loop_ordered_static_start (start, end, incr,
					     0, istart, iend);
    default:
      abort ();
    }
}

/* The *_next routines are called when the thread completes processing of 
   the iteration block currently assigned to it.  If the work-share 
   construct is bound directly to a parallel construct, then the iteration
   bounds may have been set up before the parallel.  In which case, this
   may be the first iteration for the thread.

   Returns true if there is work remaining to be performed; *ISTART and
   *IEND are filled with a new iteration block.  Returns false if all work
   has been assigned.  */

static bool
gomp_loop_static_next (long *istart, long *iend)
{
  return !gomp_iter_static_next (istart, iend);
}

static bool
gomp_loop_dynamic_next (long *istart, long *iend)
{
  bool ret;

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_dynamic_next (istart, iend);
#else
  struct gomp_thread *thr = gomp_thread ();
  gomp_mutex_lock (&thr->ts.work_share->lock);
  ret = gomp_iter_dynamic_next_locked (istart, iend);
  gomp_mutex_unlock (&thr->ts.work_share->lock);
#endif

  return ret;
}

static bool
gomp_loop_guided_next (long *istart, long *iend)
{
  bool ret;

#ifdef HAVE_SYNC_BUILTINS
  ret = gomp_iter_guided_next (istart, iend);
#else
  struct gomp_thread *thr = gomp_thread ();
  gomp_mutex_lock (&thr->ts.work_share->lock);
  ret = gomp_iter_guided_next_locked (istart, iend);
  gomp_mutex_unlock (&thr->ts.work_share->lock);
#endif

  return ret;
}

static bool
gomp_loop_trap_next (long *istart, long *iend)
{
#ifdef HAVE_SYNC_BUILTINS
  return gomp_iter_trap_next (istart, iend);
#endif
}
static bool
gomp_loop_fact_next (long *istart, long *iend)
{
#ifdef HAVE_SYNC_BUILTINS
  return gomp_iter_fact_next (istart, iend);
#endif
}

static bool
gomp_loop_tape_next (long *istart, long *iend)
{
#ifdef HAVE_SYNC_BUILTINS
  return gomp_iter_tape_next (istart, iend);
#endif
}
static bool
gomp_loop_frac_next (long *istart, long *iend)
{
#ifdef HAVE_SYNC_BUILTINS
  return gomp_iter_frac_next (istart, iend);
#endif
}
static bool
gomp_loop_wfac_next (long *istart, long *iend)
{
#ifdef HAVE_SYNC_BUILTINS
  return gomp_iter_wfac_next (istart, iend);
#endif
}
static bool
gomp_loop_bold_next (long *istart, long *iend)
{
#ifdef HAVE_SYNC_BUILTINS
  return gomp_iter_bold_next (istart, iend);
#endif
}
static bool
gomp_loop_fsc_next (long *istart, long *iend)
{
#ifdef HAVE_SYNC_BUILTINS
  return gomp_iter_fsc_next (istart, iend);
#endif
}
static bool
gomp_loop_afac_next (long *istart, long *iend)
{
#ifdef HAVE_SYNC_BUILTINS
  return gomp_iter_afac_next (istart, iend);
#endif
}

static bool
gomp_loop_binlpt_next (long *istart, long *iend)
{
  return gomp_iter_binlpt_next (istart, iend);
}

static bool
gomp_loop_srr_next (long *istart, long *iend)
{
  return gomp_iter_srr_next (istart, iend);
}
static bool
gomp_loop_rand_next (long *istart, long *iend)
{
  return gomp_iter_rand_next (istart, iend);
}

static bool
gomp_loop_fac2_next (long *istart, long *iend)
{
  return gomp_iter_fac2_next (istart, iend);
}


bool
GOMP_loop_runtime_next (long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();

  switch (thr->ts.work_share->sched)
    {
    	case GFS_STATIC:
    	case GFS_AUTO:
    	  	return gomp_loop_static_next (istart, iend);
    	case GFS_DYNAMIC:
      		return gomp_loop_dynamic_next (istart, iend);
    	case GFS_GUIDED:
      		return gomp_loop_guided_next (istart, iend);
    	case GFS_FACT:
	  	return gomp_loop_fact_next (istart, iend);
	case GFS_TAPE:
	  	return gomp_loop_tape_next (istart, iend);
	case GFS_FRAC:
	  	return gomp_loop_frac_next (istart, iend);
	case GFS_WFAC:
	  	return gomp_loop_wfac_next (istart, iend);
	case GFS_BOLD:
	  	return gomp_loop_bold_next (istart, iend);
	case GFS_TRAP:
	  	return gomp_loop_trap_next (istart, iend);
    	case GFS_FSC:
	  	return gomp_loop_fsc_next (istart, iend);
    	case GFS_AFAC:
	  	return gomp_loop_afac_next (istart, iend);
	case GFS_BINLPT:
      		return gomp_loop_binlpt_next (istart, iend);
    	case GFS_SRR:
      		return gomp_loop_srr_next (istart, iend);
    	case GFS_RAND:
      		return gomp_loop_rand_next (istart, iend);
    	case GFS_FAC2:
      		return gomp_loop_fac2_next (istart, iend);
	default:
    	  abort ();
    }
}

/* The *_ordered_*_next routines are called when the thread completes
   processing of the iteration block currently assigned to it.

   Returns true if there is work remaining to be performed; *ISTART and
   *IEND are filled with a new iteration block.  Returns false if all work
   has been assigned.  */

static bool
gomp_loop_ordered_static_next (long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();
  int test;

  gomp_ordered_sync ();
  gomp_mutex_lock (&thr->ts.work_share->lock);
  test = gomp_iter_static_next (istart, iend);
  if (test >= 0)
    gomp_ordered_static_next ();
  gomp_mutex_unlock (&thr->ts.work_share->lock);

  return test == 0;
}

static bool
gomp_loop_ordered_dynamic_next (long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  gomp_ordered_sync ();
  gomp_mutex_lock (&thr->ts.work_share->lock);
  ret = gomp_iter_dynamic_next_locked (istart, iend);
  if (ret)
    gomp_ordered_next ();
  else
    gomp_ordered_last ();
  gomp_mutex_unlock (&thr->ts.work_share->lock);

  return ret;
}

static bool
gomp_loop_ordered_guided_next (long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();
  bool ret;

  gomp_ordered_sync ();
  gomp_mutex_lock (&thr->ts.work_share->lock);
  ret = gomp_iter_guided_next_locked (istart, iend);
  if (ret)
    gomp_ordered_next ();
  else
    gomp_ordered_last ();
  gomp_mutex_unlock (&thr->ts.work_share->lock);

  return ret;
}

bool
GOMP_loop_ordered_runtime_next (long *istart, long *iend)
{
  struct gomp_thread *thr = gomp_thread ();
  
  switch (thr->ts.work_share->sched)
    {
    case GFS_STATIC:
    case GFS_AUTO:
      return gomp_loop_ordered_static_next (istart, iend);
    case GFS_DYNAMIC:
      return gomp_loop_ordered_dynamic_next (istart, iend);
    case GFS_GUIDED:
      return gomp_loop_ordered_guided_next (istart, iend);
    default:
      abort ();
    }
}

/* The GOMP_parallel_loop_* routines pre-initialize a work-share construct
   to avoid one synchronization once we get into the loop.  */

static void
gomp_parallel_loop_start (void (*fn) (void *), void *data,
			  unsigned num_threads, long start, long end,
			  long incr, enum gomp_schedule_type sched,
			  long chunk_size, unsigned int flags)
{
  struct gomp_team *team;

  num_threads = gomp_resolve_num_threads (num_threads, 0);
  team = gomp_new_team (num_threads);
  gomp_loop_init (&team->work_shares[0], start, end, incr, sched, chunk_size, num_threads);
  gomp_team_start (fn, data, num_threads, flags, team);
}

void
GOMP_parallel_loop_static_start (void (*fn) (void *), void *data,
				 unsigned num_threads, long start, long end,
				 long incr, long chunk_size)
{
  gomp_parallel_loop_start (fn, data, num_threads, start, end, incr,
			    GFS_STATIC, chunk_size, 0);
}

void
GOMP_parallel_loop_dynamic_start (void (*fn) (void *), void *data,
				  unsigned num_threads, long start, long end,
				  long incr, long chunk_size)
{
  gomp_parallel_loop_start (fn, data, num_threads, start, end, incr,
			    GFS_DYNAMIC, chunk_size, 0);
}

void
GOMP_parallel_loop_guided_start (void (*fn) (void *), void *data,
				 unsigned num_threads, long start, long end,
				 long incr, long chunk_size)
{
  gomp_parallel_loop_start (fn, data, num_threads, start, end, incr,
			    GFS_GUIDED, chunk_size, 0);
}

void
GOMP_parallel_loop_runtime_start (void (*fn) (void *), void *data,
				  unsigned num_threads, long start, long end,
				  long incr)
{
  struct gomp_task_icv *icv = gomp_icv (false);
  gomp_parallel_loop_start (fn, data, num_threads, start, end, incr,
			    icv->run_sched_var, icv->run_sched_modifier, 0);
}

ialias_redirect (GOMP_parallel_end)

void
GOMP_parallel_loop_static (void (*fn) (void *), void *data,
			   unsigned num_threads, long start, long end,
			   long incr, long chunk_size, unsigned flags)
{
  gomp_parallel_loop_start (fn, data, num_threads, start, end, incr,
			    GFS_STATIC, chunk_size, flags);
  fn (data);
  GOMP_parallel_end ();
}

void
GOMP_parallel_loop_dynamic (void (*fn) (void *), void *data,
			    unsigned num_threads, long start, long end,
			    long incr, long chunk_size, unsigned flags)
{
  gomp_parallel_loop_start (fn, data, num_threads, start, end, incr,
			    GFS_DYNAMIC, chunk_size, flags);
  fn (data);
  GOMP_parallel_end ();
}

void
GOMP_parallel_loop_guided (void (*fn) (void *), void *data,
			  unsigned num_threads, long start, long end,
			  long incr, long chunk_size, unsigned flags)
{
  gomp_parallel_loop_start (fn, data, num_threads, start, end, incr,
			    GFS_GUIDED, chunk_size, flags);
  fn (data);
  GOMP_parallel_end ();
}

void
GOMP_parallel_loop_runtime (void (*fn) (void *), void *data,
			    unsigned num_threads, long start, long end,
			    long incr, unsigned flags)
{
  struct gomp_task_icv *icv = gomp_icv (false);
  gomp_parallel_loop_start (fn, data, num_threads, start, end, incr,
			    icv->run_sched_var, icv->run_sched_modifier,
			    flags);
  fn (data);
  GOMP_parallel_end ();
}

/* The GOMP_loop_end* routines are called after the thread is told that
   all loop iterations are complete.  The first two versions synchronize
   all threads; the nowait version does not.  */

void
GOMP_loop_end (void)
{
  gomp_work_share_end ();
}


bool
GOMP_loop_end_cancel (void)
{
  return gomp_work_share_end_cancel ();
}

void
GOMP_loop_end_nowait (void)
{
  gomp_work_share_end_nowait ();
}


/* We use static functions above so that we're sure that the "runtime"
   function can defer to the proper routine without interposition.  We
   export the static function with a strong alias when possible, or with
   a wrapper function otherwise.  */

#ifdef HAVE_ATTRIBUTE_ALIAS
extern __typeof(gomp_loop_static_start) GOMP_loop_static_start
	__attribute__((alias ("gomp_loop_static_start")));
extern __typeof(gomp_loop_dynamic_start) GOMP_loop_dynamic_start
	__attribute__((alias ("gomp_loop_dynamic_start")));
extern __typeof(gomp_loop_guided_start) GOMP_loop_guided_start
	__attribute__((alias ("gomp_loop_guided_start")));

extern __typeof(gomp_loop_ordered_static_start) GOMP_loop_ordered_static_start
	__attribute__((alias ("gomp_loop_ordered_static_start")));
extern __typeof(gomp_loop_ordered_dynamic_start) GOMP_loop_ordered_dynamic_start
	__attribute__((alias ("gomp_loop_ordered_dynamic_start")));
extern __typeof(gomp_loop_ordered_guided_start) GOMP_loop_ordered_guided_start
	__attribute__((alias ("gomp_loop_ordered_guided_start")));

extern __typeof(gomp_loop_static_next) GOMP_loop_static_next
	__attribute__((alias ("gomp_loop_static_next")));
extern __typeof(gomp_loop_dynamic_next) GOMP_loop_dynamic_next
	__attribute__((alias ("gomp_loop_dynamic_next")));
extern __typeof(gomp_loop_guided_next) GOMP_loop_guided_next
	__attribute__((alias ("gomp_loop_guided_next")));

extern __typeof(gomp_loop_ordered_static_next) GOMP_loop_ordered_static_next
	__attribute__((alias ("gomp_loop_ordered_static_next")));
extern __typeof(gomp_loop_ordered_dynamic_next) GOMP_loop_ordered_dynamic_next
	__attribute__((alias ("gomp_loop_ordered_dynamic_next")));
extern __typeof(gomp_loop_ordered_guided_next) GOMP_loop_ordered_guided_next
	__attribute__((alias ("gomp_loop_ordered_guided_next")));
#else
bool
GOMP_loop_static_start (long start, long end, long incr, long chunk_size,
			long *istart, long *iend)
{
  return gomp_loop_static_start (start, end, incr, chunk_size, istart, iend);
}

bool
GOMP_loop_dynamic_start (long start, long end, long incr, long chunk_size,
			 long *istart, long *iend)
{
  return gomp_loop_dynamic_start (start, end, incr, chunk_size, istart, iend);
}

bool
GOMP_loop_guided_start (long start, long end, long incr, long chunk_size,
			long *istart, long *iend)
{
  return gomp_loop_guided_start (start, end, incr, chunk_size, istart, iend);
}

bool
GOMP_loop_ordered_static_start (long start, long end, long incr,
				long chunk_size, long *istart, long *iend)
{
  return gomp_loop_ordered_static_start (start, end, incr, chunk_size,
					 istart, iend);
}

bool
GOMP_loop_ordered_dynamic_start (long start, long end, long incr,
				 long chunk_size, long *istart, long *iend)
{
  return gomp_loop_ordered_dynamic_start (start, end, incr, chunk_size,
					  istart, iend);
}

bool
GOMP_loop_ordered_guided_start (long start, long end, long incr,
				long chunk_size, long *istart, long *iend)
{
  return gomp_loop_ordered_guided_start (start, end, incr, chunk_size,
					 istart, iend);
}

bool
GOMP_loop_static_next (long *istart, long *iend)
{
  return gomp_loop_static_next (istart, iend);
}

bool
GOMP_loop_dynamic_next (long *istart, long *iend)
{
  return gomp_loop_dynamic_next (istart, iend);
}

bool
GOMP_loop_guided_next (long *istart, long *iend)
{
  return gomp_loop_guided_next (istart, iend);
}

bool
GOMP_loop_ordered_static_next (long *istart, long *iend)
{
  return gomp_loop_ordered_static_next (istart, iend);
}

bool
GOMP_loop_ordered_dynamic_next (long *istart, long *iend)
{
  return gomp_loop_ordered_dynamic_next (istart, iend);
}

bool
GOMP_loop_ordered_guided_next (long *istart, long *iend)
{
  return gomp_loop_ordered_guided_next (istart, iend);
}
#endif
