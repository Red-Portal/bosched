
#include "GP.hpp"
#include "acquisition.hpp"

#include <dlib/global_optimization.h>

extern "C"
{
    double const beta = 2;

    double next_point(double* x, double* y, int n, int iter, int max_iter)
    {
        auto model = lpbo::gp_model (lpbo::vec(n, x), lpbo::vec(n, y)); 
        double annealing = 0.5;

        auto f = [&](double x){
                     auto [mean, var] = model.predict(x);
                     return lpbo::UCB(mean, var, beta, annealing, iter);
                 };
        auto result = dlib::find_min_global(
               f, {0.0}, {1.0}, dlib::max_function_calls(max_iter));
        return result.x;
    }

    void render_acquisition(double* x, double* y, int n, int iter,
                            double* x_hat, double* y_hat, int m)
    {
        auto model = lpbo::gp_model (lpbo::vec(n, x), lpbo::vec(n, y)); 
        double annealing = 0.5;

        auto f = [&](double x){
                     auto [mean, var] = model.predict(x);
                     return lpbo::UCB(mean, var, beta, annealing, iter);
                 };

        for(size_t i = 0; i < size_t(m); ++i)
        {
            y_hat[i] = f(x_hat[i]);
        }
    }
}
