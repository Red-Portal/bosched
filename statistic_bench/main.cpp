
#include <iostream>

#define STATS_DONT_USE_OPENMP
#define STATS_USE_BLAZE

#include "stats/include/stats.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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
             Dist dist,
             Rng& rng)
        : _runtimes(num_tasks)
    {
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

    inline size_t
    num_tasks() const
    {
        return _runtimes.size();
    }
};

template<typename Dist, typename Rng, typename Fun>
class generate_biased
{
    std::vector<duration_t> _runtimes;

public:
    inline
    generate_biased(size_t num_tasks,
                    Dist dist,
                    Rng& rng,
                    Fun bias_f)
        : _runtimes(num_tasks)
    {
        for(size_t i = 0; i < num_tasks; ++i)
        {
            auto runtime = dist(rng) + bias_f(i);
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

    inline size_t
    num_tasks() const
    {
        return _runtimes.size();
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

template<typename Gen>
void benchmark(Gen&& tasks,
               std::string_view name,
               double dist_mean,
               double dist_stddev)
{
    using clock = std::chrono::steady_clock;
    using duration_t = std::chrono::duration<double, std::milli>;

    size_t num_tasks = tasks.num_tasks();
    size_t iteration = 64;

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

    auto mu    = mean(measures.begin(), measures.end(), 0.0);
    auto sigma = stddev(measures.begin(), measures.end(), mu);
    auto conf  = sigma * 1.96 / sqrt(iteration);
    std::cout << name << ',' << mu << ',' << conf << ','
              << dist_mean << ',' << dist_stddev << std::endl;
}

int main()
{
    auto seed = std::random_device();
    auto rng = std::mt19937_64(seed());

    std::cout << "distribution,mean,+-,dist_mean,dist_stddev"  << std::endl;

    {
        auto warmup =
            [](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 1;
                double const sigma = 1;
                do
                    {
                        value = stats::rnorm(mu, sigma, rng);   
                    } while(value < 0.0);
                return value;
            };
        auto gen = generate(1024 * 32, warmup, rng);
        benchmark(gen, "gaussian1", 10, 1);
    }

    {
        auto gaussian =
            [](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 2;
                double const sigma = 0.1;
                do
                {
                    value = stats::rnorm(mu, sigma, rng);   
                } while(value < 0.0);
                return value;
            };
        auto gen = generate(1024 * 32, gaussian, rng);
        benchmark(gen, "gaussian1", 10, 1);
    }

    {
        auto gaussian =
            [](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 2;
                double const sigma = 0.5;
                do
                {
                    value = stats::rnorm(mu, sigma, rng);   
                } while(value < 0.0);
                return value;
            };
        auto gen = generate(1024 * 32, gaussian, rng);
        benchmark(gen, "gaussian2", 10, 1);
    }

    {
        auto dist =
            [](std::mt19937_64& rng)
            {
                double value;
                double const mu    = 2;
                double const sigma = 0.1;
                do
                {
                    value = stats::rnorm(mu, sigma, rng);   
                } while(value < 0.0);
                return value;
            };
        auto gen = generate(1024 * 4, dist, rng);
        benchmark(gen, "gaussian3", 10, 1);
    }

    {
        auto dist =
            [](std::mt19937_64& rng)
            {
                double const length = 0.5;
                return stats::rexp(length, rng);   
            };
        auto gen = generate(1024 * 32, dist, rng);
        benchmark(gen, "exponential1", 10, 1);
    }

    {
        auto dist =
            [](std::mt19937_64& rng)
            {
                double const length = 1;
                return stats::rexp(length, rng);   
            };
        auto gen = generate(1024 * 32, dist, rng);
        benchmark(gen, "exponential2", 10, 1);
    }

    {
        auto dist =
            [](std::mt19937_64& rng)
            {
                double const length = 0.5;
                return stats::rexp(length, rng);   
            };
        auto gen = generate(1024 * 4, dist, rng);
        benchmark(gen, "exponential3", 10, 1);
    }

    {
        auto gaussian =
            [](std::mt19937_64& rng)
            {
                double const mu    = 0.0;
                double const sigma = 0.1;
                return stats::rnorm(mu, sigma, rng);   
            };
        size_t ntasks = 1024 * 16;
        auto gen = generate_biased(ntasks, gaussian, rng,
                                   [ntasks](size_t i) {
                                       return -static_cast<double>(i) / ntasks + 3;
                                   });
        benchmark(gen, "gaussian_biased1", 10, 1);
    }

    {
        auto gaussian =
            [](std::mt19937_64& rng)
            {
                double const mu    = 0.0;
                double const sigma = 0.1;
                return stats::rnorm(mu, sigma, rng);   
            };

        size_t ntasks = 1024 * 16;
        auto gen = generate_biased(ntasks, gaussian, rng,
                                   [ntasks](size_t i) {
                                       return static_cast<double>(i) / ntasks + 1;
                                   });
        benchmark(gen, "gaussian_biased2", 10, 1);
    }

    {
        auto gaussian =
            [](std::mt19937_64& rng)
            {
                double const mu    = 0.0;
                double const sigma = 0.1;
                return stats::rnorm(mu, sigma, rng);   
            };

        size_t ntasks = 1024 * 16;
        auto gen = generate_biased(ntasks, gaussian, rng,
                                   [ntasks](size_t i) {
                                       if(i < 128 || ntasks - 128 < i)
                                           return 1.0;
                                       else
                                           return 2.0;
                                   });
        benchmark(gen, "gaussian_biased3", 10, 1);
    }
}
