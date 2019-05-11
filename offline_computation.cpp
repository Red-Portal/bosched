
#include <algorithm>
#include <boost/program_options.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
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
        (void)ntasks;
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
        auto taskmap = std::vector<unsigned>(ntasks);

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

template<typename It, typename Float>
inline Float
mean(It begin, It end, Float init)
{
    int n = end - begin;
#pragma  omp parallel for reduction(+:init) schedule(static)
    for(int i = 0; i < n; ++i)
    {
        init += *begin;
    }
    return init / n;
}


template<typename It, typename Float>
inline Float
stddev(Float mean, It begin, It end)
{
    Float sum = 0;
    int n     = (end - begin);
#pragma  omp parallel for reduction(+:sum) schedule(static)
    for(int i = 0; i < n; ++i)
    {
        Float temp = (*begin - mean);
        sum += temp * temp;
    }
    return sqrt(sum / (n - 1));
}

inline std::vector<float>
iteration_mean(nlohmann::json const& loop)
{
    size_t num_iters   = loop[0].size();
    size_t num_samples = loop.size();
    auto means = std::vector<float>(num_iters);

#pragma  omp parallel for schedule(static)
    for(int i = 0; i < static_cast<int>(num_iters); ++i)
    {
        auto buffer = std::vector<float>(num_samples);

        for(size_t j = 0; j < num_samples; ++j)
            buffer[j] = loop[j][i];
        std::nth_element(buffer.begin(),
                         buffer.begin() + num_samples / 2,
                         buffer.end());
        // double sum = 0;
        // for(size_t j = 0; j < num_samples; ++j)
        // {
        //     double elem = loop[j][i];
        //     sum += elem;
        // }
        // means[i] = sum / num_samples;
        means[i] = buffer[num_samples / 2];
    }
    return means;
}

inline std::vector<unsigned>
quantize(std::vector<float>&& loop)
{
    auto sum       = std::accumulate(loop.begin(), loop.end(), 0.0f);
    auto max_value = std::numeric_limits<unsigned>::max();

    auto T   = [max_value, sum](float elem){
                   auto scaled = elem / 10.0f * (max_value / sum);
                   return static_cast<unsigned>(
                       std::min(scaled, static_cast<float>(max_value)));
               };
    auto result = std::vector<unsigned>(loop.size());
#pragma  omp parallel for schedule(static)
    for(int i = 0; i < static_cast<int>(loop.size()); ++i)
    {
        result[i] = T(loop[i]);
    }
    return result;
}


int main(int argc, char** argv)
{
    namespace po = boost::program_options; 
    using namespace std::literals::string_literals;

    po::options_description desc("Options"); 
    desc.add_options() 
        ("path", po::value<std::string>()->default_value("."), "path") 
        ("chunks", po::value<unsigned>()->default_value(32), "binlpt chunks")
        ("threads", po::value<unsigned>()->default_value(32), "threads")
        ("h", po::value<double>(), "overhead in microsecond");

    po::variables_map vm;
    try 
    { 
        po::store(po::command_line_parser(argc, argv)
                  .options(desc).run(), vm);
        po::notify(vm); 
    } 
    catch(po::error& e) 
    { 
      std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
      std::cerr << desc << std::endl; 
      return 1;
    } 

    double h         = vm["h"].as<double>();
    auto path        = vm["path"].as<std::string>();
    unsigned chunks  = vm["chunks"].as<unsigned>();
    unsigned threads = vm["threads"].as<unsigned>();

    auto loops       = nlohmann::json();
    auto prof_stream = std::ifstream(path + "/.workload.json"s);
    if(!prof_stream)
        throw std::runtime_error("cannot find workload.json!");
    else
        prof_stream >> loops;


    nlohmann::json output;
    for(auto& [key, value] : loops.items())
    {
        std::cout << "Loop: " << key << '\n'
                  << " - iterations: " << value[0].size() << '\n'
                  << " - samples: " << value.size() << '\n'
                  << std::endl;

        auto means = iteration_mean(value);
        auto mu    = mean(means.begin(), means.end(), 0.0f);
        auto sigma = stddev(mu, means.begin(), means.end());

        auto quantized = quantize(std::move(means));
        auto taskmap = binlpt::binlpt_balance(
            quantized.data(), quantized.size(), threads, chunks);

        double css_param = h / sigma;
        double fss_param = sigma / mu;

        auto& loop_output     = output[key];
        loop_output["css"]    = css_param;
        loop_output["fss"]    = fss_param;
        loop_output["hss"]    = std::move(quantized);
        loop_output["binlpt"] = std::move(taskmap);


        std::cout << " - mean: " << mu << '\n'
                  << " - sdev: " << sigma << '\n'
                  << " - css:  " << css_param << '\n'
                  << " - fss:  " << fss_param << '\n'
                  << std::endl;
    }
    auto out_stream = std::ofstream(path + "/.params.json"s);
    out_stream << output.dump(2);
    return 0;
}
