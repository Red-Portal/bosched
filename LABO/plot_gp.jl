
@recipe function f(gp::TimeMarginalizedGP; β=0.95, obsv=true, var=false)
    nmgp = gp.non_marg_gp.gp[1]
    xlims --> (minimum(nmgp.x[1,:]), maximum(nmgp.x[1,:]))
    ylims --> (minimum(nmgp.x[2,:]), maximum(nmgp.x[2,:]))
    xmin, xmax = plotattributes[:xlims]
    ymin, ymax = plotattributes[:ylims]
    x = range(xmin, stop=xmax, length=20)
    y = range(ymin, stop=ymax, length=20)
    xgrid = repeat(x', 20, 1)
    ygrid = repeat(y, 1, 20)
    μ, Σ = predict_y(gp,[vec(xgrid)';vec(ygrid)'])
    if var
        zgrid  = reshape(Σ,20,20)
    else
        zgrid  = reshape(μ,20,20)
    end
    x, y, zgrid
end
