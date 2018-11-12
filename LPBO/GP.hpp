#ifndef _LPBO_GP_HPP_
#define _LPBO_GP_HPP_

#include <blaze/Blaze.h>
#include <iostream>
#include <chrono>
#include <cassert>
#include <ifstream>
//#include <blaze/math/lapack/posv.h>
#include "MCMC.hpp"

namespace lpbo
{
    using vec = blaze::DynamicVector<double, blaze::columnVector>;
    using mat = blaze::DynamicMatrix<double, blaze::columnMajor>;
    using lower = blaze::LowerMatrix<lpbo::mat>;
    using id_mat = blaze::IdentityMatrix<double>;

    class gp_model
    {
        size_t _n;
        double _ker_l; 
        double _ker_g; 
        double _variance;
        double _mean;
        lpbo::mat _K_inv;
        lpbo::lower _L;
        lpbo::vec _alpha;
        lpbo::vec _data_x;

        inline double
        covariance_kernel(double x,
                          double y,
                          double l,
                          double var) const noexcept
        {
            auto temp = (x - y) / l;
            return var * exp(-0.5 * temp * temp);
        }

        inline lpbo::mat
        covariance_kernel(lpbo::vec const& x,
                          double l,
                          double var) const noexcept
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
        covariance_kernel(double x,
                          lpbo::vec const& v,
                          double l,
                          double var) const noexcept
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
        mean(lpbo::vec const& v) const noexcept
        {
            return blaze::sum(v) / v.size();
        }

        inline lpbo::vec
        sub(lpbo::vec const& v, double a) const noexcept
        {
            auto ret = lpbo::vec(v);
            for(size_t i = 0; i < v.size(); ++i)
                ret[i] = v[i] - a;
            return ret;
        }

        inline double
        loglikelihood(lpbo::lower const& L,
                      lpbo::vec const& alpha,
                      lpbo::vec const& data_y) const noexcept
        {
            auto diag = blaze::band(L, 0l);
            return (-0.5) * blaze::dot(data_y, alpha)
                - blaze::sum(blaze::log(diag))
                - 0.5 * log(2 * 3.14159265359);
        }

        inline lpbo::lower
        compute_inverse(lpbo::vec const& x,
                        double l,
                        double g,
                        double var,
                        size_t n) const noexcept
        {
            auto K = covariance_kernel(x, l, var);
            K += g * lpbo::id_mat(n);

            auto buff = lpbo::mat(n, n);
            blaze::llh(K, buff);
            return lpbo::lower(buff);
        }

    public:
        inline
        gp_model() noexcept
        {}

        inline void
        matrix_deserialize(blaze::Archive<std::ifstream>& archive)
        {
            ;
        }

        inline
        gp_model(lpbo::vec const& x, lpbo::vec const& y)
        {
            _data_x = x;
            _mean = mean(y);
            auto y_norm = sub(y, _mean);
            _variance = blaze::dot(y_norm, y_norm) / (y_norm.size() - 1);
            _n = x.size();

            auto f = [&](lpbo::vec param){
                         auto L = compute_inverse(x,
                                                  param[0],
                                                  param[1],
                                                  _variance,
                                                  _n);
                         auto alpha = y_norm;
                         blaze::trsv(lpbo::mat(L), alpha, 'L', 'N', 'N');
                         blaze::trsv(lpbo::mat(L), alpha, 'L', 'T', 'N');
                         return loglikelihood(L, alpha, y_norm);
                     };

            auto param = metropolis_hastings(f, lpbo::vec({0.01, 0.01}), 1000);
            _ker_l = param[0];
            _ker_g = param[1];

            std::cout << param << std::endl;
            _L = compute_inverse(x, _ker_l, _ker_g, _variance, _n);

            auto L_inv = _L;
            blaze::invert(L_inv);
            _K_inv = blaze::trans(L_inv) * L_inv;
            _alpha = _K_inv * y_norm;
        }

        inline std::pair<double, double>
        predict(double x)
        {
            auto k = covariance_kernel(x, _data_x,  _ker_l, _variance);
            auto mu = blaze::dot(k, _alpha) + _mean;

            auto sigma2 = covariance_kernel(x, x, _ker_l, _variance)
                - blaze::trans(k) * (_K_inv * k);
            return {mu, sigma2};
        }
    };
}

#endif
