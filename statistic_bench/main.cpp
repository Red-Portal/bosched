
#define STATS_DONT_USE_OPENMP
#define STATS_USE_BLAZE

#include "stats/include/stats.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

template <class Tp>
inline void do_not_optimize(Tp const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
inline void do_not_optimize(Tp& value) {
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

std::random_device _seed;
std::mt19937_64 _rng(_seed());

using duration_t = std::chrono::duration<double, std::micro>;

template<typename Dist, typename Rng>
class generate
{
    std::vector<duration_t> _runtimes;

public:
    inline
    generate(size_t num_tasks,
             size_t threads,
             Dist dist,
             Rng& rng)
        : _runtimes(num_tasks)
    {
        auto workers = std::vector<std::thread>();
        workers.reserve(threads);

        for(size_t i = 0; i < threads; ++i)
        {
            size_t begin = (num_tasks / threads) * i;
            size_t end = (num_tasks / threads) * (i + 1);
            end = std::min(num_tasks, end);
            auto seed = rng();
            auto work =
                [&]
                {
                    auto local_rng = std::mt19937_64(seed);
                    for(size_t i = begin; i < end; ++i)
                    {
                        auto runtime = dist(local_rng);
                        _runtimes[i] = duration_t(runtime); 
                    }
                };
            workers.emplace_back(work);
        }

        for(auto& i : workers)
            i.join();
    }

    inline duration_t
    operator[](size_t i)
    {
        return _runtimes[i];
    }

    inline duration_t
    operator[](int i)
    {
        return _runtimes[i];
    }
};

template<typename Type, typename It>
Type
mean(It begin, It end, Type init)
{
    Type sum = std::accumulate(begin, end, init);
    return sum / (end - begin);
}

template<typename Type, typename It>
Type
stddev(It begin, It end, Type mean)
{
    Type sum = std::accumulate(begin, end, Type(),
                               [mean](Type sum, Type elem){
                                   Type temp = elem - mean;
                                   return sum + temp * temp;
                               });
    return sqrt(sum) / (end - begin - 1);
}


template<typename Dist, typename Rng>
void benchmark(Rng& rng,
               Dist dist,
               size_t num_tasks,
               std::string_view name,
               double dist_mean,
               double dist_stddev)
{
    using clock = std::chrono::steady_clock;
    using duration_t = std::chrono::duration<double, std::milli>;

    auto duration = duration_t();
    auto tasks = generate<Dist, Rng>(num_tasks, 8, dist, rng);
    size_t iteration = 128;

    if(getenv("EVAL"))
    {
        auto measures = std::vector<double>(num_tasks);
        for(size_t it = 0; it < iteration; ++it)
        {
            auto loop_start = clock::now();
#pragma omp parallel for schedule(runtime)
            for(size_t i = 0; i < num_tasks; ++i)
            {
                auto start = clock::now();
                auto runtime = tasks[i];
                auto now = clock::now();
                while(now - start < runtime)
                {
                    now = clock::now();
                    do_not_optimize(now);
                }
            }
            auto loop_end = clock::now();
            measures[it] = std::chrono::duration_cast<duration_t>(
                loop_end - loop_start).count();
        }
        auto mu = mean(measures.begin(), measures.end(), 0.0);
        auto sigma = stddev(measures.begin(), measures.end(), mu);
        std::cout << name << ',' << mu << ',' << sigma << ','
                  << dist_mean << ',' << dist_stddev << std::endl;
    }
    else
    {
        auto loop_start = clock::now();
#pragma omp parallel for schedule(runtime)
        for(size_t i = 0; i < num_tasks; ++i)
        {
            auto start = clock::now();
            auto runtime = tasks[i];
            auto now = clock::now();
            while(now - start < runtime)
            {
                now = clock::now();
                do_not_optimize(now);
            }
        }
        auto loop_end = clock::now();
        duration = std::chrono::duration_cast<duration_t>(
            loop_end - loop_start);
        std::cout << name << ',' << duration.count() << ','
                  << dist_mean << ',' << dist_stddev << std::endl;
    }
}

int main()
{
    auto seed = std::random_device();
    auto rng = std::mt19937_64(seed());

    {
        double const p = 0.5;
        double const val = 1;
        auto bern1 =
            [p, val](std::mt19937_64& rng)
            {
                return val * stats::rbern(p, rng);
            };
        benchmark(rng, bern1, 1024 * 1024, "bern1", 0.5, 0.5);
    }

    {
        double const p = 0.2;
        double const val = 2.5;
        auto bern2 =
            [p, val](std::mt19937_64& rng)
            {
                return val * stats::rbern(p, rng);
            };
        benchmark(rng, bern2, 1024 * 1024, "bern2", 0.2, 2.5);
    }

    {
        double const p = 0.8;
        double const val = 10.0/8.0;
        auto bern3 =
            [](std::mt19937_64& rng)
            {
                double const val = 1;
                double const p = 0.5;
                return val * stats::rbern(p, rng);
            };
        benchmark(rng, bern3, 1024 * 1024, "bern3", 1.0, 0.5);
    }

    {
        auto uniform1 =
            [](std::mt19937_64& rng)
            {
                double const val = 1;
                double const a = 0;
                double const b = 1;
                return val * stats::runif(a, b, rng);
            };
        benchmark(rng, uniform1, 1024 * 1024, "uniform1", 0.5, 0.28867513459481287);
    }

    {
        auto uniform2 =
            [](std::mt19937_64& rng)
            {
                double const val = 1;
                double const a = 0.3;
                double const b = 0.7;
                return val * stats::runif(a, b, rng);
            };
        benchmark(rng, uniform2, 1024 * 1024, "uniform2", 0.5, 0.11547005383792512);
    }

    {
        auto uniform3 =
            [](std::mt19937_64& rng)
            {
                double const val = 2;
                double const a = 0.4;
                double const b = 0.6;
                return val * stats::runif(a, b, rng);
            };
        benchmark(rng, uniform3, 1024 * 1024, "uniform3", 1.0, 0.11547005383792512);
    }

    {
        auto poisson1 =
            [](std::mt19937_64& rng)
            {
                double const rate = 2;
                double const value = 0.25;
                return value * stats::rpois(rate, rng);
            };
        benchmark(rng, poisson1, 1024 * 1024, "poisson1", 0.5, 0.7071067811865476);
    }

    {
        auto poisson2 =
            [](std::mt19937_64& rng)
            {
                double const rate = 0.125;
                double const value = 4;
                return value * stats::rpois(rate, rng);
            };
        benchmark(rng, poisson2, 1024 * 1024, "poisson2", 0.5, 1.4142135623730951);
    }

    {
        auto poisson3 =
            [](std::mt19937_64& rng)
            {
                double const rate = 10;
                double const value = 0.1;
                return value * stats::rpois(rate, rng);
            };
        benchmark(rng, poisson3, 1024 * 1024, "poisson3", 1.0, 0.316227766016838);
    }

    {
        auto gaussian1 =
            [](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 0.407347;
                double const sigma = 0.369224;
                do
                {
                    value = stats::rnorm(mu, sigma, rng);   
                } while(value < 0.0);
                return value;
            };
        benchmark(rng, gaussian1, 1024 * 1024, "gaussian1", 0.5, 0.3);
    }

    {
        auto gaussian2 =
            [](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 0.5;
                double const sigma = 0.1;
                do
                {
                    value = stats::rnorm(mu, sigma, rng);   
                } while(value < 0.0);
                return value;
            };
        benchmark(rng, gaussian2, 1024 * 1024, "gaussian2", 0.5, 0.1);
    }

    {
        auto gaussian3 =
            [](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 0.999519;
                double const sigma = 0.3008;
                do
                {
                    value = stats::rnorm(mu, sigma, rng);   
                } while(value < 0.0);
                return value;
            };
        benchmark(rng, gaussian3, 1024 * 1024, "gaussian3", 1.0, 0.3);
    }
}
