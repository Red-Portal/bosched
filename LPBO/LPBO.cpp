
#include <cstdio>
#include <dlib/global_optimization.h>
#include <string>

#include "SMC.hpp"
#include "LPBO.hpp"
#include "acquisition.hpp"

extern "C"
{
    void* test_init(double* x, double* y, int n)
    {
        auto* model = new lpbo::smc_gp(lpbo::vec(n, x), lpbo::vec(n, y), 10, 0.9); 
        return reinterpret_cast<void*>(model);
    }

    void test_render_acquisition(void* model, int iter,
                                 int resolution, double* y_hat)
    {
        auto* gp_model = reinterpret_cast<lpbo::smc_gp*>(model);
        auto y = lpbo::render_acquisition(*gp_model, iter, resolution);
        for(size_t i = 0; i < static_cast<size_t>(resolution); ++i)
        {
            y_hat[i] = y[i];
        }
    }

    void test_render_gp(void* model, int resolution,
                        double* means, double* vars)
    {
        auto* gp_model = reinterpret_cast<lpbo::smc_gp*>(model);
        auto [means_hat, vars_hat] = lpbo::render_gp(*gp_model, resolution);
        std::copy(means_hat.begin(), means_hat.end(), means);
        std::copy(vars_hat.begin(),  vars_hat.end(),  vars);
    }

    double test_next_point(void* model, double x, double y,
                           int iter, int max_iter,
                           double* out_merit,
                           double* out_predmean,
                           double* out_predvar)
    {
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

    void test_get_history(void* model, double* x, double* y,  int* n)
    {
        auto* gp_model = reinterpret_cast<lpbo::smc_gp*>(model);
        auto const& data_x = gp_model->data_x();
        auto const& data_y = gp_model->data_y();
        std::copy(data_x.begin(), data_x.end(), x);
        std::copy(data_y.begin(), data_y.end(), y);
        *n = gp_model->data_x().size();
        std::cout << *n << std::endl;
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
