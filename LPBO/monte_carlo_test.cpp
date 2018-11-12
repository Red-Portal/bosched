
#include <iostream>
#include <chrono>
//#include "SMC.hpp"
#include "MCMC.hpp"

double function(double x, size_t)
{
    return lpbo::normal_pdf(x, 0.05, 0.01);
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
        auto result = lpbo::metropolis_hastings(function2, 500);
        auto stop = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<
            std::chrono::duration<double, std::milli>>(stop - start).count();
        std::cout << "result " << result << '\n'
                  << duration << "ms" << std::endl;
    }
}
