
#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <string>
#include <ctime>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cstdlib>

namespace bosched
{
    inline std::string
    format_current_time() noexcept
    {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];

        time (&rawtime);
        timeinfo = localtime(&rawtime);

        strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
        return std::string(buffer);
    }

	
/* @brief Sets the workload of the next parallel for loop.	
 *	
 * @param loop_id     The ID of the loop to attach workload information to.	
 * @param tasks       Load of iterations.	
 * @param ntasks      Number of tasks.	
 * @param override    Boolean flag to decide whether we should compute the	
 *                    task mapping again or use the preexisting one.		
 */

    // inline void omp_set_workload(unsigned loop_id,	
    // 				 unsigned *tasks,	
    // 				 unsigned ntasks,	
    // 				 bool override)	
    // {	
    // 	/* Make sure the loop id is correct. */
    // 	assert((0 <= loop_id) && (loop_id < R_LOOPS));	

    // 	/* Make sure omp_set_workload() is not called in parallel. */
    // 	assert(gomp_thread()->ts.team_id == 0);	

    // 	loops[loop_id].override = override;	
    // 	curr_loop = loop_id;	

    // 	__tasks = tasks;	
    // 	__ntasks = ntasks;	
    // }	

#define exch(a, b)                              \
  do {                                          \
    unsigned tmp = (a);                         \
    (a) = (b);                                  \
    (b) = tmp;                                  \
  } while(0);	

    inline void	
    insertion(unsigned* map, unsigned* a, unsigned n)	
    {	
	unsigned i, j; /* Loop indexes.    */	

	/* Sort. */	
	for (i = 0; i < (n - 1); i++)	
	{	
	    for (j = i + 1; j < n; j++)	
	    {	
		/* Swap. */	
		if (a[j] < a[i])	
		{	
		    exch(a[i], a[j]);	
		    exch(map[i], map[j]);	
		}	
	    }	
	}	
    }	

    void quicksort(unsigned *map, unsigned *a, unsigned n)	
    {	
	unsigned i, j;	
	unsigned p;	

	/* End recursion. */	
	if (n < 128)	
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
	    exch(a[i], a[j]);	
	    exch(map[i], map[j]);	
	}	

	quicksort(map, a, i);	
	quicksort(map, a + i, n - i);	
    }	

    void sort(unsigned *a,	
	      unsigned n,	
	      unsigned *map)	
    {	
	unsigned i;	

	/* Create map. */	
	for (i = 0; i < n; i++)	
	    map[i] = i;	

	insertion(map, a, n);	
    }	

    inline unsigned*	
    compute_cummulativesum(unsigned const* a,	
			   unsigned n)	
    {	
	unsigned i;	
	unsigned *sum;	

	sum = static_cast<unsigned*>(malloc(n*sizeof(unsigned)));	
	assert(sum != NULL);	

	for (sum[0] = 0, i = 1; i < n; i++)	
	    sum[i] = sum[i - 1] + a[i - 1];	

	return (sum);	
    }	

    inline unsigned*	
    compute_chunksizes(unsigned const* tasks,	
		       unsigned ntasks,	
		       unsigned nchunks)	
    {	
	unsigned i, k;	
	unsigned chunkweight;	
	unsigned *chunksizes, *workload;	

	chunksizes = static_cast<unsigned*>(calloc(nchunks, sizeof(unsigned)));	
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
    inline unsigned*
    compute_chunks(unsigned const* tasks,	
		   unsigned ntasks,	
		   unsigned const* chunksizes,	
		   unsigned nchunks)	
    {	
	unsigned i, k;    /* Loop indexes. */	
	unsigned *chunks; /* Chunks.       */	

	chunks = static_cast<unsigned*>(calloc(nchunks, sizeof(unsigned)));	
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
    inline void
    binlpt_balance(unsigned const* tasks,	
		   unsigned ntasks,	
		   unsigned nthreads,
		   unsigned max_chunks,
		   unsigned* taskmap)
    {	
	unsigned i;               /* Loop index.       */	
	unsigned *chunksizes;     /* Chunks sizes.     */	
	unsigned *chunks;         /* Chunks.           */	
	unsigned *chunkoff;       /* Offset to chunks. */	

	/* Initialize scheduler data. */	
	assert(taskmap != NULL);	

	auto sortmap = std::vector<unsigned>(ntasks, 0); /* Sorting map.       */	
	auto load    = std::vector<size_t>(nthreads, 0.0); /* Assigned load.    */	

	//printf("[binlpt] Balancing loop %s:%i\n", loops[curr_loop].filename, loops[curr_loop].line);	

	chunksizes = compute_chunksizes(tasks, ntasks, max_chunks);	
	chunks     = compute_chunks(tasks, ntasks, chunksizes, max_chunks);	
	chunkoff   = compute_cummulativesum(chunksizes, max_chunks);	

	/* Sort tasks. */	
	sort(chunks, max_chunks, sortmap.data());	

	for (i = max_chunks; i > 0; i--)	
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
	    load[tid] += chunks[sortmap[i - 1]];	
	}	

	for (size_t i = 0; i < ntasks; ++i)
	    printf("%d\n", taskmap[i]);
	printf("----------------\n");

	for (i = max_chunks; i > 0; i--)	
	    printf("%d\n", chunks[sortmap[i - 1]]);

	printf("----------------\n");

	for (size_t i = 0; i < nthreads; ++i)
	    printf("%ld\n", load[i]);


	/* House keeping. */	
	free(chunkoff);	
	free(chunks);	
	free(chunksizes);	
    }
}

#endif
