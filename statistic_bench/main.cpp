
#include <iostream>

#define STATS_DONT_USE_OPENMP
#define STATS_USE_BLAZE

#include "stats/include/stats.hpp"

#include <algorithm>
#include <chrono>
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
        // auto workers = std::vector<std::thread>();
        // workers.reserve(threads);

        // size_t portion = ceil(static_cast<double>(num_tasks) / threads);
        // for(size_t i = 0; i < threads; ++i)
        // {
        //     size_t begin = portion * i;
        //     size_t end = std::min(portion * (i + 1), num_tasks);
        //     auto seed = rng();
        //     auto work =
        //         [&]
        //         {
        //             auto local_rng = std::mt19937_64(seed);
        //             for(size_t i = begin; i < end; ++i)
        //             {
        //                 auto runtime = dist(local_rng);
        //                 _runtimes[i] = duration_t(runtime); 
        //             }
        //         };
        //     workers.emplace_back(work);
        // }

        // for(auto& i : workers)
        //     i.join();

        for(size_t i = 0; i < num_tasks; ++i)
        {
            auto runtime = dist(rng);
            _runtimes[i] = duration_t(runtime); 
        }
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
    return sqrt(sum / (end - begin - 1));
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

    auto tasks = generate<Dist, Rng>(num_tasks, 8, dist, rng);
    size_t iteration = 64;

    if(getenv("EVAL"))
    {
        iteration = 1024;
    }

    auto measures = std::vector<double>(iteration);
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

int main()
{
    auto seed = std::random_device();
    auto rng = std::mt19937_64(seed());

    std::cout << "distribution,mean,stddev,dist_mean,dist_stddev"  << std::endl;

    {
        auto warmup =
            [](std::mt19937_64& rng)
            {
                double const p = 0.99;
                double const val = 1000.0 / 99;
                return val * stats::rbern(p, rng);
            };
        benchmark(rng, warmup, 1024 * 32, "warmup", 10, 1.0050378152592125);
    }

    {
        auto bern1 =
            [](std::mt19937_64& rng)
            {
                double const p = 0.99;
                double const val = 1000.0 / 99;
                return val * stats::rbern(p, rng);
            };
        benchmark(rng, bern1, 1024 * 32, "bern1", 10, 1.0050378152592125);
    }

    {
        auto bern2 =
            [](std::mt19937_64& rng)
            {
                double const p = 0.8;
                double const val = 1000.0 / 80;
                return val * stats::rbern(p, rng);
            };
        benchmark(rng, bern2, 1024 * 32, "bern2", 10, 5);
    }

    {
        auto bern3 =
            [](std::mt19937_64& rng)
            {
                double const p = 0.999;
                double const val = 50000.0 / 999;
                return val * stats::rbern(p, rng);
            };
        benchmark(rng, bern3, 1024 * 32, "bern3", 50, 1.5819299929208324);
    }

    {
        auto uniform1 =
            [](std::mt19937_64& rng)
            {
                double const val = 2;
                double const a = 4;
                double const b = 6;
                return val * stats::runif(a, b, rng);
            };
        benchmark(rng, uniform1, 1024 * 32, "uniform1", 10, 1.1547005383792515);
    }

    {
        auto uniform2 =
            [](std::mt19937_64& rng)
            {
                double const val = 2;
                double const a = 0;
                double const b = 10;
                return val * stats::runif(a, b, rng);
            };
        benchmark(rng, uniform2, 1024 * 32, "uniform2", 10, 5.773502691896258);
    }

    {
        auto uniform3 =
            [](std::mt19937_64& rng)
            {
                double const val = 1;
                double const a = 48;
                double const b = 52;
                return val * stats::runif(a, b, rng);
            };
        benchmark(rng, uniform3, 1024 * 32, "uniform3", 50, 1.1547005383792515);
    }

    {
        auto poisson1 =
            [](std::mt19937_64& rng)
            {
                double const rate = 50;
                double const value = 0.2;
                return value * stats::rpois(rate, rng);
            };
        benchmark(rng, poisson1, 1024 * 32, "poisson1", 10, 1.4142135623730951);
    }

    {
        auto poisson2 =
            [](std::mt19937_64& rng)
            {
                double const rate = 2;
                double const value = 5;
                return value * stats::rpois(rate, rng);
            };
        benchmark(rng, poisson2, 1024 * 32, "poisson2", 10, 7.0710678118654755);
    }

    {
        auto poisson3 =
            [](std::mt19937_64& rng)
            {
                double const rate = 50;
                double const value = 1;
                return value * stats::rpois(rate, rng);
            };
        benchmark(rng, poisson3, 1024 * 32, "poisson3", 50, 7.0710678118654755);
    }

    {
        auto gaussian1 =
            [](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 10;
                double const sigma = 1;
                do
                {
                    value = stats::rnorm(mu, sigma, rng);   
                } while(value < 0.0);
                return value;
            };
        benchmark(rng, gaussian1, 1024 * 32, "gaussian1", 10, 1);
    }

    {
        auto gaussian2 =
            [](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 9.686285744350245;
                double const sigma = 5;
                do
                {
                    value = stats::rnorm(mu, sigma, rng);   
                } while(value < 0.0);
                return value;
            };
        benchmark(rng, gaussian2, 1024 * 32, "gaussian2", 9.686285744350245, 5);
    }

    {
        auto gaussian3 =
            [](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 50;
                double const sigma = 1;
                do
                {
                    value = stats::rnorm(mu, sigma, rng);   
                } while(value < 0.0);
                return value;
            };
        benchmark(rng, gaussian3, 1024 * 32, "gaussian3", 50, 1);
    }
}
