
#include <cstdio>
#include "SMC.hpp"
#include "acquisition.hpp"

#include <dlib/global_optimization.h>

double const beta = 2;

extern "C"
{
    void* test_init(double* x, double* y, int n)
    {
        auto* model = new lpbo::smc_gp(10, lpbo::vec(n, x), lpbo::vec(n, y)); 
        return reinterpret_cast<void*>(model);
    }

    void test_render_acquisition(void* model, double* x, double* y, int n, int iter,
                                 double* x_hat, double* y_hat, int m)
    {
        double annealing = 0.5;
        auto* gp_model = reinterpret_cast<lpbo::smc_gp*>(model);

        auto f = [&](double x){
                     auto [mean, var] = gp_model->predict(x);
                     return lpbo::UCB(mean, var, beta, annealing, iter);
                 };

        for(size_t i = 0; i < size_t(m); ++i)
        {
            y_hat[i] = f(x_hat[i]);
        }
    }

    double test_next_point(void* model, double x, double y,
                           int iter, int max_iter,
                           double* out_merit,
                           double* out_predmean,
                           double* out_predvar)
    {
        double annealing = 0.5;

        auto* gp_model = reinterpret_cast<lpbo::smc_gp*>(model);
        if(iter > 1)
            gp_model->update(x, y);

        auto f = [&](double x){
                     auto [mean, var] = gp_model->predict(x);
                     return lpbo::UCB(mean, var, beta, annealing, iter);
                 };

        auto result = dlib::find_min_global(
            f, {0.0}, {1.0}, dlib::max_function_calls(max_iter), dlib::FOREVER, 1e-3);

        auto [m, v] = gp_model->predict(result.x);

        if(out_merit != nullptr)
            *out_merit = lpbo::UCB(m, v, beta, annealing, iter);

        if(out_predmean != nullptr)
            *out_predmean = m;

        if(out_predvar != nullptr)
            *out_predvar = v;
        return result.x;
    }

}
