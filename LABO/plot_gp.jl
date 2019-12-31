
@recipe function f(gp::GaussianProcesses.GPBase; β=0.95, obsv=true, var=false)
    @assert gp.dim ∈ (1,2)
    if gp.dim == 1
        xlims --> (minimum(gp.x), maximum(gp.x))
        xmin, xmax = plotattributes[:xlims]
        x = range(xmin, stop=xmax, length=100)
        x = reshape(x, (1, :))
        μ, Σ = predict_f(gp, x)
        y = μ
        err = norminvcdf((1+β)/2)*sqrt.(Σ)
        
        @series begin
            seriestype := :path
            ribbon := err
            fillcolor --> :lightblue
            color --> :black
            x',y
        end
        if obsv
            @series begin
                seriestype := :scatter
                markershape := :circle
                markercolor := :black
                gp.x', gp.y
            end
        end
    else
        xlims --> (minimum(gp.x[1,:]), maximum(gp.x[1,:]))
        ylims --> (minimum(gp.x[2,:]), maximum(gp.x[2,:]))
        xmin, xmax = plotattributes[:xlims]
        ymin, ymax = plotattributes[:ylims]
        x = range(xmin, stop=xmax, length=50)
        y = range(ymin, stop=ymax, length=50)
        xgrid = repeat(x', 50, 1)
        ygrid = repeat(y, 1, 50)
        μ, Σ = predict_f(gp,[vec(xgrid)';vec(ygrid)'])
        if var
            zgrid  = reshape(Σ,50,50)
        else
            zgrid  = reshape(μ,50,50)
        end
        x, y, zgrid
    end
end
