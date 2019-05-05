
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vector>

#define N 128

#define exch(a, b)                              \
    do {                                        \
        unsigned tmp = (a);                     \
        (a) = (b);                              \
        (b) = tmp;                              \
    } while(0);

namespace binlpt
{
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

        sum = (unsigned*)malloc(n*sizeof(unsigned));
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

        chunksizes = (unsigned*)calloc(nchunks, sizeof(unsigned));
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

        chunks = (unsigned*)calloc(nchunks, sizeof(unsigned));
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

    inline std::vector<unsigned>
    binlpt_balance(unsigned* tasks,
                   unsigned ntasks,
                   unsigned nthreads,
                   unsigned nchunks)
    {
        unsigned i;               /* Loop index.       */
        //unsigned *taskmap;        /* Task map.         */
        unsigned sortmap[ntasks]; /* Sorting map.       */
        unsigned *load;           /* Assigned load.    */
        unsigned *chunksizes;     /* Chunks sizes.     */
        unsigned *chunks;         /* Chunks.           */
        unsigned *chunkoff;       /* Offset to chunks. */

        //printf("[binlpt] Balancing loop %s:%i\n", loops[curr_loop].filename, loops[curr_loop].line);

        /* Initialize scheduler data. */
        // taskmap = (unsigned*)calloc(ntasks, sizeof(unsigned));
        auto taskmap = std::vector<unsigned>(ntasks, static_cast<unsigned>(0));

        load    = (unsigned*)calloc(nthreads, sizeof(unsigned));
        assert(load != NULL);

        chunksizes = compute_chunksizes(tasks, ntasks, nchunks);
        chunks = compute_chunks(tasks, ntasks, chunksizes, nchunks);
        chunkoff = compute_cummulativesum(chunksizes, nchunks);

        /* Sort tasks. */
        sort(chunks, nchunks, sortmap);

        for (i = nchunks; i > 0; i--)
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
        return taskmap;
    }
}

template<typename It>
double mean(It begin, It end)
{
    double sum = 0;
    for(It begin; begin != end; ++begin)
    {
        sum += *begin;
    }
    return sum / (end - begin);
}


template<typename It>
double stdddev(double mean, It begin, It end)
{
    double sum = 0;
    for(It begin; begin != end; ++begin)
    {
        double temp = (*begin - mean);
        sum += temp * temp;
    }
    size_t n = (end - begin);
    return sqrt(sum / (n - 1));
}

nlohmann::json
process_loop(nlohmann::json const& loop)
{
    auto means = std::vector<double>(loop.size());
    for(auto& iter: loop)
    {
        auto begin = iter.cbegin();
        auto end   = iter.cend();
        auto mu    = mean(begin, end);
        //auto sd    = stddev(mu, begin, end);
        means[]
            }
}

int main(int argc, char** argv)
{
    (void)argc, (void)argv;
    using namespace std::literals::string_literals;

    auto loops = nlohmann::json({1, 2, 3, 4});
    auto path = std::string(argv[1]);
    auto prof_stream = std::ifstream(path + "/workload_prof.json"s);

    for(auto& [key, value] : loops)
    {
        process_loop(value);
    }


    auto taskmap = binlpt_balance(unsigned* tasks,
                                  unsigned ntasks,
                                  unsigned nthreads,
                                  unsigned nchunks);
}
