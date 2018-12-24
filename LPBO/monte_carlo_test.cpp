
#include <iostream>
#include <chrono>
#include <cmath>
//#include "SMC.hpp"
#include "MCMC.hpp"
#include "GP.hpp"

template<typename Type, typename Mat>
double log_normal_pdf(Type x, Type mean, Mat var) noexcept
{
    auto det = blaze::det(var);
    auto temp = (x - mean);
    return -0.5 * log(det) + (-0.5) * blaze::trans(temp) * lpbo::mat{{1000, 0}, {0, 10000}} * temp;
}

double function(lpbo::vec const& x)
{
    return log_normal_pdf(x, lpbo::vec({0.01, 0.001}), lpbo::mat{{0.1, 0}, {0, 0.01}});
}

double function2(double x)
{
    return lpbo::normal_pdf(x, 0.05, 0.01);
}

int main()
{
    /*{
      auto start = std::chrono::steady_clock::now();
      auto results = lpbo::sequential_monte_carlo(function, 0.1, 10, 3, 5000);
      auto stop = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<
      std::chrono::duration<double, std::milli>>(stop - start).count();

      double weight = 0.0;
      double result = 0.0;
      for(auto const& i : results)
      {
      std::cout << "value: " << i.first << " weigth " << i.second << std::endl;
      result += i.first * i.second;
      weight += i.second;
      }
      std::cout << "result " << result / weight << '\n'
                  << duration << "ms" << std::endl;
                  }*/

    {
        auto start = std::chrono::steady_clock::now();
        auto result = lpbo::metropolis_hastings(function, lpbo::vec{0.01, 0.01}, 100);
        auto stop = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<
            std::chrono::duration<double, std::milli>>(stop - start).count();
        std::cout << "result " << result << '\n'
                  << duration << "ms" << std::endl;
    }
}
