
#ifndef _LPBO_LPBO_HPP_
#define _LPBO_LPBO_HPP_

extern "C"
{
    double next_point(double* x, double* y, int n, int iter, int max_iter,
                      double* out_predmean,  double* out_predvar);

    void render_acquisition(double* x, double* y, int n, int iter,
                            double* x_hat, double* y_hat, int m);
}

#endif
