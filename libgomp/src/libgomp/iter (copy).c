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

/* This file contains routines for managing work-share iteration, both
   for loops and sections.  */
#define _GNU_SOURCE
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
/* This function implements the STATIC scheduling method.  The caller should
   iterate *pstart <= x < *pend.  Return zero if there are more iterations
   to perform; nonzero if not.  Return less than 0 if this thread had
   received the absolutely last iteration.  */
#ifndef max
	#define max(a,b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
	#define min(a,b) ((a) < (b) ? (a) : (b))
#endif
double diff(struct timespec start, struct timespec end);	  

int
gomp_iter_static_next (long *pstart, long *pend)
{
  struct gomp_thread *thr = gomp_thread ();
  struct gomp_team *team = thr->ts.team;
  struct gomp_work_share *ws = thr->ts.work_share;
  unsigned long nthreads = team ? team->nthreads : 1;

  if (thr->ts.static_trip == -1)
    return -1;

  /* Quick test for degenerate teams and orphaned constructs.  */
  if (nthreads == 1)
    {
      *pstart = ws->next;
      *pend = ws->end;
      thr->ts.static_trip = -1;
      return ws->next == ws->end;
    }

  /* We interpret chunk_size zero as "unspecified", which means that we
     should break up the iterations such that each thread makes only one
     trip through the outer loop.  */
  if (ws->chunk_size == 0)
    {
      unsigned long n, q, i, t;
      unsigned long s0, e0;
      long s, e;

      if (thr->ts.static_trip > 0)
	return 1;

      /* Compute the total number of iterations.  */
      s = ws->incr + (ws->incr > 0 ? -1 : 1);
      n = (ws->end - ws->next + s) / ws->incr;
      i = thr->ts.team_id;

      /* Compute the "zero-based" start and end points.  That is, as
         if the loop began at zero and incremented by one.  */
      q = n / nthreads;
      t = n % nthreads;
      if (i < t)
	{
	  t = 0;
	  q++;
	}
      s0 = q * i + t;
      e0 = s0 + q;

      /* Notice when no iterations allocated for this thread.  */
      if (s0 >= e0)
	{
	  thr->ts.static_trip = 1;
	  return 1;
	}

      /* Transform these to the actual start and end numbers.  */
      s = (long)s0 * ws->incr + ws->next;
      e = (long)e0 * ws->incr + ws->next;

      *pstart = s;
      *pend = e;
      thr->ts.static_trip = (e0 == n ? -1 : 1);
      return 0;
    }
  else
    {
      unsigned long n, s0, e0, i, c;
      long s, e;

      /* Otherwise, each thread gets exactly chunk_size iterations
	 (if available) each time through the loop.  */

      s = ws->incr + (ws->incr > 0 ? -1 : 1);
      n = (ws->end - ws->next + s) / ws->incr;
      i = thr->ts.team_id;
      c = ws->chunk_size;

      /* Initial guess is a C sized chunk positioned nthreads iterations
	 in, offset by our thread number.  */
      s0 = (thr->ts.static_trip * nthreads + i) * c;
      e0 = s0 + c;

      /* Detect overflow.  */
      if (s0 >= n)
	return 1;
      if (e0 > n)
	e0 = n;

      /* Transform these to the actual start and end numbers.  */
      s = (long)s0 * ws->incr + ws->next;
      e = (long)e0 * ws->incr + ws->next;

      *pstart = s;
      *pend = e;

      if (e0 == n)
	thr->ts.static_trip = -1;
      else
	thr->ts.static_trip++;
      return 0;
    }
}


/* This function implements the DYNAMIC scheduling method.  Arguments are
   as for gomp_iter_static_next.  This function must be called with ws->lock
   held.  */

bool
gomp_iter_dynamic_next_locked (long *pstart, long *pend)
{
  struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  long start, end, chunk, left;

  start = ws->next;
  if (start == ws->end)
    return false;

  chunk = ws->chunk_size;
  left = ws->end - start;
  if (ws->incr < 0)
    {
      if (chunk < left)
	chunk = left;
    }
  else
    {
      if (chunk > left)
	chunk = left;
    }
  end = start + chunk;

  ws->next = end;
  *pstart = start;
  *pend = end;
  return true;
}


#ifdef HAVE_SYNC_BUILTINS
/* Similar, but doesn't require the lock held, and uses compare-and-swap
   instead.  Note that the only memory value that changes is ws->next.  */

bool
gomp_iter_dynamic_next (long *pstart, long *pend)
{
  struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  long start, end, nend, chunk, incr;

  end = ws->end;
  incr = ws->incr;
  chunk = ws->chunk_size;

  if (__builtin_expect (ws->mode, 1))
    {
      long tmp = __sync_fetch_and_add (&ws->next, chunk);
      if (incr > 0)
	{
	  if (tmp >= end)
	    return false;
	  nend = tmp + chunk;
	  if (nend > end)
	    nend = end;
	  *pstart = tmp;
	  *pend = nend;
	  return true;
	}
      else
	{
	  if (tmp <= end)
	    return false;
	  nend = tmp + chunk;
	  if (nend < end)
	    nend = end;
	  *pstart = tmp;
	  *pend = nend;
	  return true;
	}
    }

  start = ws->next;
  while (1)
    {
      long left = end - start;
      long tmp;

      if (start == end)
	return false;

      if (incr < 0)
	{
	  if (chunk < left)
	    chunk = left;
	}
      else
	{
	  if (chunk > left)
	    chunk = left;
	}
      nend = start + chunk;

      tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
      if (__builtin_expect (tmp == start, 1)){
	printf("%ld dyn , thrNr: %i, end: %ld,  nend: %ld\n", start, omp_get_thread_num(), end,  nend);
	break;
	}

      start = tmp;
    }

  *pstart = start;
  *pend = nend;
  return true;
}

/*int ipow(int base, int exp)
{
    int result = 1;
	    while (exp)
	    {
	        if (exp & 1) result *= base;
			exp >>= 1;
	        base *= base;
	    }
    return result;
}

int iround(double x)
{
    if (x < 0.0)
        return (int)(x - 0.5);
    else
        return (int)(x + 0.5);
}*/

//This is probably very unsafe. 
extern unsigned __ntasks; 



bool
gomp_iter_fact_next (long *pstart, long *pend)
{
  //printf("hi");

  struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  struct gomp_team *team = thr->ts.team;
  unsigned long nthreads = team ? team->nthreads : 1;
  long start, end, nend;// incr;
  unsigned long chunk_size;
  start = ws->next;
  end = ws->end;
  //incr = ws->incr;
  chunk_size = ws->chunk_size;
  //if (globalfactcounter == -1) maxworkload = (end-start);///incr;
  int mycounter =__sync_add_and_fetch(&ws->globalfactcounter, 1);
  /*if (mycounter == 12) { 
 void *array[10];
   size_t size;
     char **strings;
	   size_t i;
	   
	     size = backtrace (array, 10);
		   strings = backtrace_symbols (array, size);
		   
		     printf ("Obtained %zd stack frames.\n", size);
			 
			   for (i = 0; i < size; i++)
			        printf ("%s\n", strings[i]);
					
					  free (strings);
  }*/
  while (1)
    {
      unsigned long n, q;
      long tmp;

      if (start == end)
	return false;
      n = (end - start); /// incr;
	  //if (globalfactcounter%nthreads == 0) {
	  int blatest = (mycounter/nthreads)+1;
	  q = ceil((double) ws->maxworkload /(pow(2, blatest)* nthreads)); 
	  //q = q >>blatest;
	//__sync_fetch_and_add(&globalfactcounter, 1);
	//q = (n + nthreads - 1) / (2*nthreads);
	//currentworkload = q;
	  //}
	  //else {
	//__sync_fetch_and_add(&globalfactcounter, 1);
	//q = currentworkload;
	  //}
      if (q < chunk_size)
	q = chunk_size;
      if (__builtin_expect (q <= n, 1))
	nend = start + q;
      else
	nend = end;

      tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
      if (__builtin_expect (tmp == start, 1)){

//printf("cpu: %d, thrn: %d\n",sched_getcpu(), omp_get_thread_num());
//printf("%ld fact ,q: %ld, n: %ld, thrNr: %i, end: %ld,  nend: %ld\n", start, q, n, omp_get_thread_num(), end,  nend);
break;}
	  //printf("\nhi\n");
	 // __sync_sub_and_fetch(&globalfactcounter, 1);
      start = tmp;
    }
 // printf("threadid: %d, start %ld, end %ld\n", omp_get_thread_num(), start, end);

  *pstart = start;
  *pend = nend;
  return true;
}
//This is probably very unsafe. 
/*int globalfactcounter = -1;
int currentworkload = 0;

extern unsigned __ntasks; 
bool
gomp_iter_fact_next (long *pstart, long *pend)
{
  struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  struct gomp_team *team = thr->ts.team;
  unsigned long nthreads = team ? team->nthreads : 1;
  long start, end, nend, incr;
  unsigned long chunk_size;
  start = ws->next;
  end = ws->end;
  incr = ws->incr;
  chunk_size = ws->chunk_size;
  
  while (1)
    {
      unsigned long n, q;
      long tmp;

      if (start == end)
	return false;
	__sync_fetch_and_add(&globalfactcounter, 1);
      n = (end - start) / incr;
	  if (globalfactcounter%nthreads == 0) {
	//__sync_fetch_and_add(&globalfactcounter, 1);
	q = (n + nthreads - 1) / (2*nthreads);
	currentworkload = q;
	  }
	  else {
	//__sync_fetch_and_add(&globalfactcounter, 1);
	q = currentworkload;
	  }
//printf("q: %ld, n: %ld, nthreads: %ld\n", q, n, nthreads);
      if (q < chunk_size)
	q = chunk_size;
      if (__builtin_expect (q <= n, 1))
	nend = start + q * incr;
      else
	nend = end;

printf("q: %ld, n: %ld, nthreads: %ld, start: %ld, end: %ld, incr: %ld, nend: %ld\n", q, n, nthreads, start, end, incr, nend);
      tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
      if (__builtin_expect (tmp == start, 1))
	break;

      start = tmp;
    }

  *pstart = start;
  *pend = nend;
  return true;
}*/
bool
gomp_iter_frac_next (long *pstart, long *pend)
{return true;

}





bool
gomp_iter_tape_next (long *pstart, long *pend)
{
//printf("hi");
  struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  struct gomp_team *team = thr->ts.team;
  unsigned long nthreads = team ? team->nthreads : 1;
  long start, end, nend;// incr;
  unsigned long chunk_size;
  start = ws->next;
  end = ws->end;
  /*sigma = strtod(getenv("SIGMA"),NULL);
  alpha = strtod(getenv("ALPHA"),NULL);
  mue = strtod(getenv("MUE"),NULL);
  //incr = ws->incr;
  va = (alpha*sigma)/mue;*/
  //double Ki = (Ti + ((va*va)/2) - (va*sqrt(2*Ti+((va*va)/4))));
  chunk_size = ws->chunk_size;
  //if (globalfactcounter == -1) maxworkload = (end-start);///incr;
  //__sync_add_and_fetch(&globalfactcounter, 1);
  while (1)
    {
      unsigned long n, q;
      long tmp;
	double Ti = (end - start)/nthreads;
      if (start == end)
	return false;
      n = (end - start); /// incr;
	  //if (globalfactcounter%nthreads == 0) {
	  q =  (Ti + ((ws->va*ws->va)/2) - (ws->va*sqrt(2*Ti+((ws->va*ws->va)/4))));
	  //q = q >>blatest;
	//__sync_fetch_and_add(&globalfactcounter, 1);
	//q = (n + nthreads - 1) / (2*nthreads);
	//currentworkload = q;
	  //}
	  //else {
	//__sync_fetch_and_add(&globalfactcounter, 1);
	//q = currentworkload;
	  //}
      if (q < chunk_size)
	q = chunk_size;
      if (__builtin_expect (q <= n, 1))
	nend = start + q;
      else
	nend = end;

      tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
      if (__builtin_expect (tmp == start, 1)){
//printf("%ld,",q);
//printf("%ld tape ,q: %ld, n: %ld, thrNr: %i, end: %ld,  nend: %ld\n", start, q, n, omp_get_thread_num(), end,  nend);
break;}
	  //printf("\nhi\n");
	 // __sync_sub_and_fetch(&globalfactcounter, 1);
      start = tmp;
    }
 // printf("threadid: %d, start %ld, end %ld\n", omp_get_thread_num(), start, end);

  *pstart = start;
  *pend = nend;
  return true;

}





bool
gomp_iter_wfac_next (long *pstart, long *pend)
{
  //printf("hi");
  struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  struct gomp_team *team = thr->ts.team;
  unsigned long nthreads = team ? team->nthreads : 1;
  long start, end, nend, incr;
  unsigned long chunk_size;
  start = ws->next;
  end = ws->end;
  incr = ws->incr;
  int tempcounter =__sync_add_and_fetch(&ws->globalWFACcounter, 1);

   //printf("/n%f/n",asdf[0]);
  chunk_size = ws->chunk_size;

 while (1)
    {

      unsigned long n, q;
      long tmp;
	
      if (start == end)
	return false;
      n = (end - start) / incr;
	  //if (globalfactcounter%nthreads == 0) {
	  int blatest = (tempcounter/nthreads)+1;
	  q = ceil((ws->WFACarray[omp_get_thread_num()])*((ws->WFACworkload) / (pow(2,blatest) * nthreads))); 
	//__sync_fetch_and_add(&globalfactcounter, 1);
	//q = (n + nthreads - 1) / (2*nthreads);
	//currentworkload = q;
	  //}
	  //else {
	//__sync_fetch_and_add(&globalfactcounter, 1);
	//q = currentworkload;
	  //}
	  
      if (q < chunk_size)
	q = chunk_size;
      if (__builtin_expect (q <= n, 1))
	nend = start + q * incr;
      else
	nend = end;
	

      tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
      if (__builtin_expect (tmp == start, 1)){
//printf("%ld,",q);
//printf("%ld wfac ,q: %ld, n: %ld, thrNr: %i, end: %ld, incr: %ld, nend: %ld, multi: %f\n", start, q, n, omp_get_thread_num(), end, incr, nend,asdf[omp_get_thread_num()]);
break;}
	  //printf("\nhi\n");
	 // __sync_sub_and_fetch(&globalfactcounter, 1);
      start = tmp;
    }
 // printf("threadid: %d, start %ld, end %ld\n", omp_get_thread_num(), start, end);

  *pstart = start;
  *pend = nend;
  return true;
}
double diff(struct timespec start, struct timespec end)
{
	struct timespec temp;
		if ((end.tv_nsec-start.tv_nsec)<0) {
			temp.tv_sec = end.tv_sec-start.tv_sec-1;
			temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
		} else {
			temp.tv_sec = end.tv_sec-start.tv_sec;
			temp.tv_nsec = end.tv_nsec-start.tv_nsec;
		}
	double time = temp.tv_sec + (double)(temp.tv_nsec)*0.000000001;
	return time;
}













pthread_mutex_t lockm;
pthread_mutex_t lockn;
pthread_mutex_t lockspeed;
pthread_mutex_t locktime;




bool
gomp_iter_bold_next (long *pstart, long *pend)
{
struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  struct gomp_team *team = thr->ts.team;
  unsigned long nthreads = team ? team->nthreads : 1;
  long start, end, nend;// incr;
  unsigned long chunk_size;
  start = ws->next;
  end = ws->end;
  
  struct timespec boldt = ws->boldtime;

pthread_mutex_lock(&locktime);
		clock_gettime(CLOCK_MONOTONIC, &ws->boldtime);
pthread_mutex_unlock(&locktime);
 
  struct timespec boldt2;
  clock_gettime(CLOCK_MONOTONIC, &boldt2);
  
  //struct timespec boldt1;
  //clock_gettime(CLOCK_MONOTONIC, &boldt1);
  //timearray[omp_get_thread_num()] = boldt1;
  //pthread_mutex_lock(&lock);  
  //boldtime = boldt2; 
  //pthread_mutex_unlock(&lock);  
 
 //boldn -= (boldt2 - boldtime)*totalspeed+k timearray[omp_get_thread_num()] 
  
pthread_mutex_lock(&lockm);
	ws->boldm -= ws->boldarray[omp_get_thread_num()];
pthread_mutex_unlock(&lockm);
pthread_mutex_lock(&lockspeed);
	ws->totalspeed -= ws->speedarray[omp_get_thread_num()];
pthread_mutex_unlock(&lockspeed);
	   
	   
  //double Ki = (Ti + ((va*va)/2) - (va*sqrt(2*Ti+((va*va)/4))));
  chunk_size = ws->chunk_size;
  //if (globalfactcounter == -1) maxworkload = (end-start);///incr;
  //__sync_add_and_fetch(&globalfactcounter, 1);
  while (1)
    {

	  unsigned long n;
	  double q;
      long tmp;
	//double Ti = (end - start)/nthreads;
      if (start == end)
	return false;
	  n = (end - start); /// incr;
	  q = n/nthreads;
	  double r = max(n,ws->boldn);
	  double t = ws->p_inv * r;
	  double ln_Q = log(q);
	  double v = q/(ws->boldb+q);
	  double d = n/(1+(1/ln_Q)-v);
	  if(d<=ws->c2) q = t;
	  double s = ws->bolda*(log(d)-ws->c3)*(1+(ws->boldm/(r*nthreads)));
	  double w = 0;
	  if (ws->boldb > 0) w = log(v*ln_Q)+ws->ln_b;
	  else w = log(ln_Q);
	  q = min(t+max(0,ws->c1*w)+(s/2)-sqrt(s*(t+(s/4))),t);
	  //if (globalfactcounter%nthreads == 0) {
	  //q =  (Ti + ((va*va)/2) - (va*sqrt(2*Ti+((va*va)/4))));
	  //q = q >>blatest;
	//__sync_fetch_and_add(&globalfactcounter, 1);
	//q = (n + nthreads - 1) / (2*nthreads);
	//currentworkload = q;
	  //}
	  //else {
	//__sync_fetch_and_add(&globalfactcounter, 1);
	//q = currentworkload;
	  //}
      if (q < 1.0)
	q = chunk_size;
      if (__builtin_expect (q <= n, 1))
	nend = start + q;
      else
	nend = end;

      tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
      if (__builtin_expect (tmp == start, 1)){
		ws->boldarray[omp_get_thread_num()] = q;
		//struct timespec boldtime;
		//clock_gettime(CLOCK_MONOTONIC, &boldtime);
		//struct timespec boldt = boldtime;
		double diff1 = diff(boldt,boldt2);
		if (!ws->boldtset){ 
			diff1 = 0;
			ws->boldtset = true;
		}
		double diff2 = diff(ws->timearray[omp_get_thread_num()], boldt2);

pthread_mutex_lock(&lockn);
	ws->boldn -= (diff1*ws->totalspeed+q-((diff2*q)/(q*ws->boldmue+ws->boldh)));
pthread_mutex_unlock(&lockn);
		
		
		ws->totalspeed += q/((q*ws->boldmue)+ws->boldh);
		ws->speedarray[omp_get_thread_num()] = q/((q*ws->boldmue)+ws->boldh);
		//struct timespec boldt1;
		//clock_gettime(CLOCK_MONOTONIC, &boldt1);
		ws->timearray[omp_get_thread_num()] = boldt2;
//printf("%ld,",q);
//printf("%ld bold ,chunk_size: %ld, t: %f , q: %f, n: %ld, thrNr: %i, end: %ld,  nend: %ld, totalspeed: %f, boldn: %f, speedarray: %f diff1: %f, diff2: %f\n", start, chunk_size,t, q, n, omp_get_thread_num(), end,  nend, totalspeed, boldn, speedarray[omp_get_thread_num()], diff1, diff2);
break;}
	  //printf("\nhi\n");
	 // __sync_sub_and_fetch(&globalfactcounter, 1);
      start = tmp;
    }
 // printf("threadid: %d, start %ld, end %ld\n", omp_get_thread_num(), start, end);

  *pstart = start;
  *pend = nend;
  return true;


}



bool
gomp_iter_trap_next (long *pstart, long *pend)
{
  //printf("hi");
  struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  //struct gomp_team *team = thr->ts.team;
  //unsigned long nthreads = team ? team->nthreads : 1;
  long start, end, nend;// incr;
  unsigned long chunk_size;
  start = ws->next;
  end = ws->end;
  //incr = ws->incr;
  chunk_size = ws->chunk_size;
  int mynumber = __sync_add_and_fetch(&ws->trapcounter, 1);
  //unsigned long testn = 0;
	
 // printf("trapcounter: %i\n", trapcounter);
  /*if (decr_delta == 0 || startsize == 0){
  __sync_sub_and_fetch(&trapcounter, 1);
  *pstart = start;
  *pend = end; 
  return true;
  }*/
  while (1) {

	unsigned long n, q;
	long tmp;
	if (start == end)	return false;
	q = 0;
	if(ws->startsize != 0 && ws->decr_delta !=0) {
		q = ws->startsize - (ws->decr_delta * mynumber);
	}
	/*else{
		startsize = atoi(getenv("TRAPSTART"));
		int endsize = atoi(getenv("TRAPEND"));
		int big_n = (2*end)/(startsize+endsize);
	//printf("hello there from thread nr: %i\n",omp_get_thread_num()); 
		if (big_n != 1) decr_delta = (startsize - endsize)/(big_n - 1);
		q = startsize - (decr_delta * mynumber);
	}*/
	n = (end - start);// / incr;
	//printf("q: %ld, n: %ld, nthreads: %ld", q, n, nthreads);
	if (q < chunk_size)
	q = chunk_size;
	if (__builtin_expect (q <= n, 1)) nend = start + q;
	else nend = end;
	tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
	if (__builtin_expect (tmp == start, 1)) {
//printf("%ld,",q);
//printf("%i, trapcounter: %i, q: %ld, n: %ld, threadNr: %i, start: %ld, end: %ld, nend: %ld, trapstart: %i, decrdelta: %f\n", mynumber, trapcounter, q, n, omp_get_thread_num(), start, end,  nend, startsize, decr_delta);
	break;}
	start = tmp;
  }
    *pstart = start;
  *pend = nend;
  return true;
}

bool
gomp_iter_fsc_next (long *pstart, long *pend)
{
  struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  long start, end, nend, chunk, incr;

  end = ws->end;
  incr = ws->incr;
  chunk = ws->fscsize;

  if (__builtin_expect (ws->mode, 1))
    {
      long tmp = __sync_fetch_and_add (&ws->next, chunk);
      if (incr > 0)
	{
	  if (tmp >= end)
	    return false;
	  nend = tmp + chunk;
	  if (nend > end)
	    nend = end;
	  *pstart = tmp;
	  *pend = nend;
	  return true;
	}
      else
	{
	  if (tmp <= end)
	    return false;
	  nend = tmp + chunk;
	  if (nend < end)
	    nend = end;
	  *pstart = tmp;
	  *pend = nend;
	  return true;
	}
    }

  start = ws->next;
  while (1)
    {
      long left = end - start;
      long tmp;

      if (start == end)
	return false;

      if (incr < 0)
	{
	  if (chunk < left)
	    chunk = left;
	}
      else
	{
	  if (chunk > left)
	    chunk = left;
	}
      nend = start + chunk;

      tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
      if (__builtin_expect (tmp == start, 1)){
	//printf("%ld fsc , thrNr: %i, end: %ld,  nend: %ld, chunk: %ld, fscsize: %i\n", start, omp_get_thread_num(), end,  nend, chunk, fscsize );
	break;
	}

      start = tmp;
    }

  *pstart = start;
  *pend = nend;
  return true;
}


//extern double *timearray;






bool
gomp_iter_afac_next (long *pstart, long *pend)
{
  struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  struct gomp_team *team = thr->ts.team;
  unsigned long nthreads = team ? team->nthreads : 1;
  long start, end, nend;// incr;
  unsigned long chunk_size;
  start = ws->next;
  end = ws->end;
  //incr = ws->incr;
  chunk_size = ws->chunk_size;
  //if (globalfactcounter == -1) maxworkload = (end-start);///incr;
 // int mycounter =__sync_add_and_fetch(&afacounter, 1);
  double currtime = 0;
  struct timespec tps; 
  clock_gettime(CLOCK_REALTIME, &tps);
  //printf("afactimetaken: %f, nanosec: %lu\n", afactimetaken[omp_get_thread_num()], tps.tv_nsec);
  if (ws->afacitercount[omp_get_thread_num()]>0) currtime = (tps.tv_nsec) - ws->afactimetaken[omp_get_thread_num()];
  if (currtime<0) currtime = 1000000000+currtime;
  //if (afactimetaken[omp_get_thread_num()] > tps.tv_nsec) printf("wth is happening.currtime: %f time: %ld, afactime: %ld, threadnr: %i, count: %i\n",currtime, (tps.tv_nsec), afactimetaken[omp_get_thread_num()], omp_get_thread_num(), afacitercount[omp_get_thread_num()]); 
  ws->afacspeed[omp_get_thread_num()*nthreads+ws->afacitercount[omp_get_thread_num()]] = currtime;
  //printf("currtime: %f, omp_get_thread_num(): %i, afacitercount: %i, speed: %f\n",currtime, omp_get_thread_num(), afacitercount[omp_get_thread_num()], afacspeed[omp_get_thread_num()*nthreads+afacitercount[omp_get_thread_num()]]);
  /*if (mycounter == 12) { 
 void *array[10];
   size_t size;
     char **strings;
	   size_t i;
	   
	     size = backtrace (array, 10);
		   strings = backtrace_symbols (array, size);
		   
		     printf ("Obtained %zd stack frames.\n", size);
			 
			   for (i = 0; i < size; i++)
			        printf ("%s\n", strings[i]);
					
					  free (strings);
  }*/
  while (1)
    {
      unsigned long n;
	  double q;
      long tmp;
		
      if (start == end)
	return false;
      n = (end - start); /// incr;
	  //if (globalfactcounter%nthreads == 0) {
	  //int blatest = (mycounter/nthreads);
	  //printf("blatest: %i\n", blatest);
	  if (n <= 0){q = end/nthreads;}
	  else{
	  if (ws->afacitercount[omp_get_thread_num()] == 0) {
	  	if (end > nthreads*1000){
			q = 100;
		}
		else q = end/(nthreads*4);
		//afacspeed[omp_get_thread_num()*nthreads+afacitercount[omp_get_thread_num()]] = time(NULL);
	    //printf("speed: %f\n", afacspeed[omp_get_thread_num()*nthreads+afacitercount[omp_get_thread_num()]]);
	  }
	  else if(ws->afacarray[omp_get_thread_num()] == 1)
	  {
	  	q = 1;
	  }
	  else
		{
			double mu = 0;
			double sigma = 0;
			//double X = 0;
			for (int i = 0; i < (ws->afacitercount[omp_get_thread_num()]); i++) currtime += ws->afacspeed[omp_get_thread_num()*nthreads+i];
			if (ws->afacarray[omp_get_thread_num()] != 0) mu = currtime/ws->afacarray[omp_get_thread_num()];
			if (ws->afacarray[omp_get_thread_num()] == 1)  ws->afacarray[omp_get_thread_num()] = 2;
			if (mu == 0) mu = 0.01;
			//sigma = pow(((currtime*(currtime/afacarray[omp_get_thread_num()]))-(afacarray[omp_get_thread_num()]*mu*mu)/(afacarray[omp_get_thread_num()]-1.0)),1.0/2.0);
			
			ws->afacmu[omp_get_thread_num()] = mu;
			ws->afacsigma[omp_get_thread_num()] = sigma;
			double D = 0;
			double T = 0;
			for (int i = 0; i < nthreads; i++) D += ((ws->afacsigma[i]*ws->afacsigma[i])/ws->afacmu[i]);
			for (int i = 0; i < nthreads; i++) T += (1.0/ws->afacmu[i]);
			T = pow(T,-1);
			q = (D+2.0*T*n-sqrt(D*D+4.0*D*T*n))/(2.0*mu);
			if(D < 0 || T < 0 || mu <= 0) printf("D: %f, T %f, mu: %f, currtime: %f\n", D, T, mu, currtime);
			//printf("%f mu: %f, sigma: %f, D: %f, T: %f, currtime: %f, q: %f, afacarray: %f, afaciter: %i threadnr: %i \n", q, mu, sigma, D, T,currtime, q, afacarray[omp_get_thread_num()], afacitercount[omp_get_thread_num()], omp_get_thread_num());
		}
}
//q = ceil((double) maxworkload /(pow(2, blatest)* nthreads)); 
	  //q = q >>blatest;
	//__sync_fetch_and_add(&globalfactcounter, 1);
	//q = (n + nthreads - 1) / (2*nthreads);
	//currentworkload = q;
	  //}
	  //else {
	//__sync_fetch_and_add(&globalfactcounter, 1);
	//q = currentworkload;
	  //}
      if (q < chunk_size)
	q = chunk_size;
      if (__builtin_expect (q <= n, 1))
	nend = start + q;
      else
	nend = end;

      tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
      if (__builtin_expect (tmp == start, 1))
	  { 
	  	struct timespec tpe;
  		clock_gettime(CLOCK_REALTIME, &tpe);
	  	//printf("q: %f, time: %ld\n", q, tps.tv_nsec);
	    ws->afacarray[omp_get_thread_num()] = q;		
		ws->afactimetaken[omp_get_thread_num()] = (tpe.tv_nsec);
		//if (afactimetaken[omp_get_thread_num()]<0) printf("wtf\n");
		ws->afacitercount[omp_get_thread_num()] += 1;
//printf("cpu: %d, thrn: %d, nthreads: %ld\n",sched_getcpu(), omp_get_thread_num(), nthreads);
//printf("%ld afac ,q: %f, n: %ld, thrNr: %i, end: %ld,  nend: %ld\n", start, q, n, omp_get_thread_num(), end,  nend);
break;}
	  //printf("\nhi\n");
	 // __sync_sub_and_fetch(&globalfactcounter, 1);
      start = tmp;
    }
 // printf("threadid: %d, start %ld, end %ld\n", omp_get_thread_num(), start, end);

  *pstart = start;
  *pend = nend;
  
return true;
}

bool
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

bool
gomp_iter_srr_next (long *pstart, long *pend)
{
  int i;                      /* Loop index.           */
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

	ws->thread_start[tid] = i + 1;
	*pstart = ws->loop_start + i;
	*pend = ws->loop_start + i + 1;

	return (true);
}

#endif /* HAVE_SYNC_BUILTINS */

/* This function implements the GUIDED scheduling method.  Arguments are
   as for gomp_iter_static_next.  This function must be called with the
   work share lock held.  */

bool
gomp_iter_guided_next_locked (long *pstart, long *pend)
{
  struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  struct gomp_team *team = thr->ts.team;
  unsigned long nthreads = team ? team->nthreads : 1;
  unsigned long n, q;
  long start, end;

  if (ws->next == ws->end)
    return false;

  start = ws->next;
  n = (ws->end - start) / ws->incr;
  q = (n + nthreads - 1) / nthreads;

  if (q < ws->chunk_size)
    q = ws->chunk_size;
  if (q <= n)
    end = start + q * ws->incr;
  else
    end = ws->end;

  ws->next = end;
  *pstart = start;
  *pend = end;
  return true;
}

#ifdef HAVE_SYNC_BUILTINS
/* Similar, but doesn't require the lock held, and uses compare-and-swap
   instead.  Note that the only memory value that changes is ws->next.  */

bool
gomp_iter_guided_next (long *pstart, long *pend)
{
  struct gomp_thread *thr = gomp_thread ();
  struct gomp_work_share *ws = thr->ts.work_share;
  struct gomp_team *team = thr->ts.team;
  unsigned long nthreads = team ? team->nthreads : 1;
  long start, end, nend, incr;
  unsigned long chunk_size;

  start = ws->next;
  end = ws->end;
  incr = ws->incr;
  chunk_size = ws->chunk_size;

  while (1)
    {
      unsigned long n, q;
      long tmp;

      if (start == end)
	return false;

      n = (end - start) / incr;
      q = (n + nthreads - 1) / nthreads;

      if (q < chunk_size)
	q = chunk_size;
      if (__builtin_expect (q <= n, 1))
	nend = start + q * incr;
      else
	nend = end;

      tmp = __sync_val_compare_and_swap (&ws->next, start, nend);
      if (__builtin_expect (tmp == start, 1))
	break;

      start = tmp;
    }

  *pstart = start;
  *pend = nend;
  return true;
}
#endif /* HAVE_SYNC_BUILTINS */
