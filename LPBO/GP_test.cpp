
#include "GP.hpp"
#include <vector>
#include <optional>
#include <iostream>

std::optional<lpbo::gp_model> _m;

extern "C"
{
    void build(double* x, double* y, int N)
    {
        auto x_vec = lpbo::vec(N, x);
        auto y_vec = lpbo::vec(N, y);
        _m = lpbo::gp_model(x_vec, y_vec);
    }

    void predict(double x, double* mean, double* var)
    {
        auto [m, v] = _m->predict(x);
        *mean = m;
        *var  = v;
    }
}

int main()
{
    auto data_x = lpbo::vec({0, 0.2, 0.5, 0.6, 0.9});
    auto data_y = lpbo::vec({1, 3, 0.5, 0.2, 0.1});

    auto start = std::chrono::steady_clock::now();
    auto model = lpbo::gp_model(data_x, data_y);
    auto stop = std::chrono::steady_clock::now();
    std::cout << "elapsed " << std::chrono::duration_cast<
        std::chrono::duration<double, std::milli>>(stop - start).count()
              << " ms" << std::endl;

    for(auto x : {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9})
    {
        start = std::chrono::steady_clock::now();
        auto [m, v] = model.predict(x);
        std::cout << "mean: " << m << " variance: " << v << std::endl;
        stop = std::chrono::steady_clock::now();
        std::cout << "elapsed " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()
                  << " us" << std::endl;
    }
}

