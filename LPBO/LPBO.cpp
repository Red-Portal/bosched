
#include <cstdio>
#include <dlib/global_optimization.h>
#include <string>

#include "SMC.hpp"
#include "LPBO.hpp"
#include "acquisition.hpp"


double const beta = 2;

extern "C"
{
    void* test_init(double* x, double* y, int n)
    {
        auto* model = new lpbo::smc_gp(lpbo::vec(n, x), lpbo::vec(n, y), 10, 0.9); 
        return reinterpret_cast<void*>(model);
    }

    void test_render_acquisition(void* model, double* x, double* y, int n, int iter,
                                 double* x_hat, double* y_hat, int m)
    {
        double annealing = 0.5;
        auto* gp_model = reinterpret_cast<lpbo::smc_gp*>(model);

        auto f = [=](double x){
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

        auto [n, m, v, cb]
            = lpbo::bayesian_optimization(*gp_model, 1e-7, iter, max_iter);

        if(out_merit != nullptr)
            *out_merit = cb;

        if(out_predmean != nullptr)
            *out_predmean = m;

        if(out_predvar != nullptr)
            *out_predvar = v;
        return n;
    }

    void test_serialize(void* model, char* str, int* n)
    {
        auto* gp_model = reinterpret_cast<lpbo::smc_gp*>(model);
        auto serialized = gp_model->serialize();
        serialized.copy(str, serialized.size());
        *n = serialized.size();
    }

    void test_deserialize(void* model, char* str, int n)
    {
        auto serialized = std::string(str, n);
        auto* gp_model = reinterpret_cast<lpbo::smc_gp*>(model);
        gp_model->deserialize(serialized);
    }
}
