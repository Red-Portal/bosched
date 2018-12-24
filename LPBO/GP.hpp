
#ifndef _LPBO_GP_HPP_
#define _LPBO_GP_HPP_

#include <blaze/Blaze.h>
#include <iostream>
#include <chrono>
#include <cassert>
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
        double _ker_l; 
        double _ker_g; 
        double _variance;
        double _mean;
        double _likelihood;
        lpbo::mat _K_inv;
        lpbo::lower _L;
        lpbo::vec _alpha;
        lpbo::vec _data_x;
        lpbo::vec _data_y;
        lpbo::vec _data_y_normalized;

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

        inline void
        update_Kinv(double x)
        {
            auto K_x = covariance_kernel(x, x, _ker_l, _variance);
            auto k_x = covariance_kernel(x, _data_x,  _ker_l, _variance);
            auto Kinv_k_x  = blaze::evaluate(_K_inv * k_x);
            auto mu = 1 / (K_x - (blaze::trans(k_x) * Kinv_k_x));
            auto g = -1 * mu * Kinv_k_x;

            auto old_size = _K_inv.rows();
            _K_inv.resize(old_size + 1, old_size + 1, true);
            
            auto matrix_view = blaze::submatrix(_K_inv, 0u, 0u, old_size, old_size);
            matrix_view += (blaze::outer(g, g) / mu);;

            for(size_t i = 0; i < old_size; ++i)
            {
                _K_inv(old_size, i) = g[i]; 
                _K_inv(i, old_size) = g[i]; 
            }
            _K_inv(old_size, old_size) = K_x;
        }

        inline void
        gp_model_init(lpbo::vec const& x, lpbo::vec const& y,
                      lpbo::vec&& mcmc_initial,
                      size_t mcmc_iterations)
        {
            _mean = mean(y);
            _data_y_normalized = sub(y, _mean);
            _variance = blaze::dot(_data_y_normalized, _data_y_normalized)
                / (_data_y_normalized.size() - 1);
            auto n = x.size();

            auto f = [&](lpbo::vec param){
                         auto L = compute_inverse(x,
                                                  param[0],
                                                  param[1],
                                                  _variance,
                                                  n);
                         auto alpha = _data_y_normalized;
                         blaze::trsv(lpbo::mat(L), alpha, 'L', 'N', 'N');
                         blaze::trsv(lpbo::mat(L), alpha, 'L', 'T', 'N');
                         return loglikelihood(L, alpha, _data_y_normalized);
                     };

            auto param = metropolis_hastings(f, std::move(mcmc_initial), mcmc_iterations);
            _ker_l = param[0];
            _ker_g = param[1];

            _L = compute_inverse(x, _ker_l, _ker_g, _variance, n);

            auto L_inv = _L;
            blaze::invert(L_inv);
            _K_inv = blaze::trans(L_inv) * L_inv;
            _alpha = _K_inv * _data_y_normalized;

            _likelihood = loglikelihood(_L, _alpha, _data_y_normalized);
        }

    public:
        inline
        gp_model() noexcept
        {}

        inline
        gp_model(lpbo::vec const& x, lpbo::vec const& y) noexcept
        {
            _data_x = x;
            _data_y = y;
            gp_model_init(x, y, lpbo::vec{0.1, 0.1}, 1000);
        }

        inline
        gp_model(lpbo::vec const& x, lpbo::vec const& y,
                 lpbo::vec&& mcmc_initial,
                 size_t mcmc_iterations) noexcept
        {
            _data_x = x;
            _data_y = y;
            gp_model_init(x, y, std::move(mcmc_initial), mcmc_iterations);
        }

        inline void
        rejuvenate(lpbo::vec const& x, lpbo::vec const& y,
                   lpbo::vec&& mcmc_initial, size_t mcmc_iterations) noexcept
        {
            size_t old_n = _data_x.size();
            size_t add_n = x.size();
            _data_x.resize(old_n + add_n);
            blaze::subvector(_data_x, old_n, add_n) = x;

            _data_y.resize(old_n + add_n);
            blaze::subvector(_data_y, old_n, add_n) = y;

            gp_model_init(_data_x, _data_y,
                          std::move(mcmc_initial), mcmc_iterations);
        }

        inline void
        matrix_deserialize(blaze::Archive<std::ifstream>& archive)
        {
            ;
        }

        inline double
        likelihood()
        {
            return _likelihood;           
        }

        inline void
        update(double x, double y)
        {
            auto old_size = _data_x.size();

            update_Kinv(x);
            
            _data_x.resize(old_size + 1);
            _data_x[old_size] = x;

            _data_y.resize(old_size + 1);
            _data_y[old_size] = y;

            _data_y_normalized.resize(old_size + 1);
            _data_y_normalized[old_size] = y - _mean;

            _alpha = _K_inv * _data_y_normalized;
            _variance = blaze::dot(_data_y_normalized, _data_y_normalized)
                / (_data_y_normalized.size() - 1);
            _likelihood = loglikelihood(_L, _alpha, _data_y_normalized);
        }

        inline void
        update(lpbo::vec const& x, lpbo::vec const& y)
        {
            size_t old_n = _data_x.size();
            size_t add_n = x.size();
            _data_x.resize(old_n + add_n);
            blaze::subvector(_data_x, old_n, add_n) = x;

            _data_y.resize(old_n + add_n);
            blaze::subvector(_data_y, old_n, add_n) = y;

            auto y_normalized = sub(y, _mean);
            _data_y_normalized.resize(old_n + add_n);
            blaze::subvector(_data_y_normalized, old_n, add_n) = y_normalized;

            _variance = blaze::dot(_data_y_normalized, _data_y_normalized)
                / (_data_y_normalized.size() - 1);

            _L = compute_inverse(_data_x, _ker_l, _ker_g, _variance, old_n + add_n);

            auto L_inv = _L;
            blaze::invert(L_inv);
            _K_inv = blaze::trans(L_inv) * L_inv;
            _alpha = _K_inv * _data_y_normalized;
            _likelihood = loglikelihood(_L, _alpha, _data_y_normalized);
        }

        inline lpbo::vec
        parameters() const
        {
            return lpbo::vec({_ker_l, _ker_g});
        }

        inline std::pair<double, double>
        predict(double x) const
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
