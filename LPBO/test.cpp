
#include <blaze/Blaze.h>
#include <iostream>
#include <chrono>
#include <cassert>
#include <optional>
//#include <blaze/math/lapack/posv.h>

namespace lpbo
{
    using vec = blaze::DynamicVector<double, blaze::columnMajor>;
    using mat = blaze::DynamicMatrix<double, blaze::columnMajor>;
    using id_mat = blaze::IdentityMatrix<double>;

    inline double
    covariance_kernel(double x, double y, double l, double var) noexcept
    {
        auto temp = (x - y) / l;
        return var * exp(-0.5 * temp * temp);
    }

    inline lpbo::mat
    covariance_kernel(lpbo::vec const& x, double l, double var) noexcept
    {
        size_t N = x.size();
        auto K = blaze::SymmetricMatrix<lpbo::mat>(N);

        for(size_t i = 0; i < N; ++i) {
            for(size_t j = i; j < N; ++j) {
                if(i == j)
                    K(i, j) = 1;
                else
                {
                    auto temp = (x[i] - x[j]) / l;
                    K(i, j) = exp(-0.5 * temp  * temp);
                }
            }
        }
        return var * K;
    }

    inline lpbo::vec
    covariance_kernel(double x, lpbo::vec const& v, double l, double var) noexcept
    {
        size_t N = v.size();
        auto temp = lpbo::vec(N);
        for(size_t i = 0; i < N; ++i) {
            temp[i] = v[i] - x;
        }
        auto exponent = (temp * temp) / (-2 * l * l);
        auto k = blaze::map(exponent, [](double elem){
                                          return exp(elem);
                                      });
        return var * k;
    }
    
    inline decltype(auto)
    mean(lpbo::vec const& v) noexcept
    {
        return blaze::sum(v) / v.size();
    }

    inline lpbo::vec
    sub(lpbo::vec const& v, double a)
    {
        auto ret = lpbo::vec(v);
        for(size_t i = 0; i < v.size(); ++i)
            ret[i] = v[i] - a;
        return ret;
    }

    inline double
    mean(lpbo::vec const& alpha,
         lpbo::vec const& covariance)
    {
        return blaze::dot(covariance, alpha);
    }

    inline double
    variance(double variance,
             lpbo::mat const& L,
             lpbo::vec const& covariance)
    {
        lpbo::vec v = covariance;
        blaze::trsv(lpbo::mat(L), v, 'L', 'N', 'N');
        return variance - blaze::dot(v, v);
    }

    inline double
    loglikelihood(lpbo::vec const& alpha,
                  lpbo::vec const& data_y,
                  lpbo::mat const& L)
    {
        
        auto diag = blaze::band(L, 0l);
        return (-0.5) * blaze::dot(data_y, alpha)
            +  blaze::sum(blaze::log(diag))
            + (data_y.size() / 2) * log(2 * 3.14159265359);
    }

    class model_t
    {
        size_t _n;
        double _ker_l; 
        double _variance;
        double _mean;
        lpbo::mat _L; 
        lpbo::vec _alpha;
        lpbo::vec _data_x;

    public:
        inline model_t(lpbo::vec const& x, lpbo::vec const& y)
        {
            _data_x = x;
            _mean = mean(y);
            auto y_norm = sub(y, _mean);
            _variance = blaze::dot(y_norm, y_norm) / (y_norm.size() - 1);
            _n = x.size();

            auto K = covariance_kernel(x, _ker_l, _variance);
            auto noise = 1e-2;

            K += noise * lpbo::id_mat(_n);
            _L = lpbo::mat(_n, _n);
            blaze::llh(K, _L);
            assert(K == _L * blaze::trans(_L));

            auto alpha = y_norm;
            blaze::trsv(lpbo::mat(_L), alpha, 'L', 'N', 'N');
            blaze::trsv(lpbo::mat(_L), alpha, 'L', 'T', 'N');
            _alpha = std::move(alpha);
        }

        inline  std::pair<double, double>
        predict(double x)
        {
            auto k = covariance_kernel(x, _data_x,  _ker_l, _variance);
            auto mean = blaze::dot(k, _alpha) + _mean;

            blaze::trsv(lpbo::mat(_L), k, 'L', 'N', 'N');
            auto var = covariance_kernel(x, x, _ker_l, _variance) - blaze::dot(k, k);
            return {mean, var};
        }
    };

    

}

std::optional<lpbo::model> _m;

extern "C"
{
    void build(double* x, double* y, int N)
    {
        auto x_vec = lpbo::vec(N, x);
        auto y_vec = lpbo::vec(N, y);
        _m = lpbo::model(x_vec, y_vec);
    }

    void predict(double x, double* mean, double* var)
    {
        auto [m, v] = _m->predict(x);
        *mean = m;
        *var  = v;
    }
}


// int main()
// {
//     auto data_x = lpbo::vec({0, 0.2, 0.5, 0.6, 0.9});
//     auto data_y = lpbo::vec({1, 3, 0.5, 0.2, 0.1});
//     auto model = lpbo::model(data_x, data_y);

//     for(auto x : {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9})
//     {
//         auto start = std::chrono::steady_clock::now();
//         auto [m, v] = model.predict(x);
//         std::cout << "mean: " << m << " variance: " << v << std::endl;
//         auto stop = std::chrono::steady_clock::now();
//         std::cout << "elapsed " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()
//                   << " us" << std::endl;
//     }
// }

