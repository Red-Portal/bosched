
using Plots

include("../IBBO/IBBO.jl")
include("simulation.jl")

function bo_fss_transform(x)
    return 2^(11*x - 7)
end

function export_csv(fname, x, y, conf)
    open(fname, "w") do io
        println(io, "x,y,+-")
        for i = 1:length(x)
            println(io, "$(x[i]),$(y[i]),$(conf[i])")
        end
        close(io)
    end
end

function export_csv(fname, x, y)
    open(fname, "w") do io
        println(io, "x, y")
        for i = 1:length(x)
            println(io, "$(x[i]),$(y[i])")
        end
        close(io)
    end
end

function export_gmm_csv(fname, gmm::Distributions.MixtureModel)
    open(fname, "w") do io
        print(io, "m1_x,m1_y")
        for i = 2:length(gmm.components)
            print(io, ",m$(i)_x,m$(i)_y")
        end
        print(io, "\n")

        x = collect(0.0:0.01:1.0)
        for i = 1:length(x)
            y = pdf.(gmm.components[1], x[i])*probs(gmm.prior)[1]
            print(io, "$(x[i]),$(y)")
            for m = 2:length(gmm.components)
                y = pdf.(gmm.components[m], x[i])*probs(gmm.prior)[m]
                print(io, ",$(x[i]),$(y)")
            end
            print(io, "\n")
        end
        close(io)
    end
end

function export_result(d::Array{Dict,1}, path)
    for i = 1:length(d)
        px = vcat([d[t]["x"] for t = 1:i]...)
        py = vcat([d[t]["y"] for t = 1:i]...)
        μ    = mean(py)
        py .-= μ
        py ./= stdm(py, mean(py))

        if(i == 1)
            export_csv(path * "/points_$(i).csv", px, py)
        else
            export_csv(path * "/gp_$(i).csv", d[i]["gpx"], -d[i]["gpm"],
                       sqrt.(d[i]["gps"])*1.96)

            export_gmm_csv(path * "/gmm_$(i).csv", d[i]["gmm"])
            open(path * "/points_$(i).csv", "w") do io
                queries = length(d[i]["x"])
                println(io, "x,y,qx,qy")
                for j = 1:(length(px)-queries)
                    if(j <= queries)
                        println(io, "$(px[j]),$(py[j]),",
                                "$(px[end-queries+j]), $(py[end-queries+j])")
                    else
                        println(io, "$(px[j]),$(py[j])")
                    end
                end
                close(io)
            end
            export_csv(path * "/acq_$(i).csv", d[i]["ax"], d[i]["ay"])
        end
    end
end

function ibbo_bayesched(prng, max_subsample)
    Random.seed!(prng)
    df = DataFrame()
    axis1 = collect(4:0.2:10)
    axis2 = collect(-7:0.5:10)

    μ    = 1.0 
    σ    = 1.0
    dist = TruncatedNormal(μ, σ, 0.0, Inf);

    P = 16
    h = 0.1
    N = 2^14
    T = 32

    subsample = min(max_subsample, T)

    log = Dict[]
    x = zeros(Float64, T)
    y = zeros(Float64, T)

    dataset_x = Float64[]
    dataset_y = Float64[]
    
    init_d = Uniform(0.0, 1.0)
    for i = 1:T
        θ      = rand(prng, init_d)
        params = Dict{Symbol, Any}(:param=>bo_fss_transform(θ));
        time, slow, speed, effi, cov = simulate(
            BO_FSS, prng, dist, P, N, h, params)
        x[i] = θ
        y[i] = time
    end

    x_sub = x[1:subsample]
    y_sub = y[1:subsample]
    push!(dataset_x, x_sub...)
    push!(dataset_y, y_sub...)
    push!(log, Dict("x"=>x_sub, "y"=>y_sub, "ir"=>minimum(dataset_y)))

    for i = 1:32
        w, μ, σ, H, αx, αy, gpx, gpμ, gpσ, samples, best_θ, best_y =
            IBBO_log(dataset_x, dataset_y, true, true)
        gmm = MixtureModel(Normal, collect(zip(μ, σ)), w)       

        for t = 1:T
            θ = begin
                result = 0.0
                while(true)
                    result = rand(prng, gmm)
                    if(result >= 0.0 && result <= 1.0)
                        break
                    end
                end
                result
            end
            params = Dict{Symbol, Any}(:param=>bo_fss_transform(θ));
            time, slow, speed, effi, cov = simulate(
                BO_FSS, prng, dist, P, N, h, params)
           x[t] = θ
            y[t] = time
        end
        x_sub = x[1:subsample]
        y_sub = y[1:subsample]
        push!(dataset_x, x_sub...)
        push!(dataset_y, y_sub...)

        for t = 1:T
            params = Dict{Symbol, Any}(:param=>bo_fss_transform(best_θ));
            time, slow, speed, effi, cov = simulate(
                BO_FSS, prng, dist, P, N, h, params)
            y[t] = time
        end
        push!(log, Dict("x"=>x_sub, "y"=>y_sub, "gmm"=>gmm, "H"=>H,
                        "ax"=>αx, "ay"=>αy, "gpx"=>gpx, "gpm"=>gpμ,
                        "gps"=>gpσ, "samples"=>samples,
                        "ir"=>minimum(dataset_y),
                        "sr"=>mean(y)))
    end
    return log
end


function mes_bayesched(prng, max_subsample)
    Random.seed!(prng)
    df = DataFrame()
    axis1 = collect(4:0.2:10)
    axis2 = collect(-7:0.5:10)

    μ    = 1.0 
    σ    = 1.0
    dist = TruncatedNormal(μ, σ, 0.0, Inf);

    P = 16
    h = 0.1
    N = 2^14
    T = 32

    subsample = min(max_subsample, T)

    log = Dict[]
    x = zeros(Float64, T)
    y = zeros(Float64, T)

    dataset_x = Float64[]
    dataset_y = Float64[]
    
    init_d = Uniform(0.0, 1.0)
    for i = 1:T
        θ      = rand(prng, init_d)
        params = Dict{Symbol, Any}(:param=>bo_fss_transform(θ));
        time, slow, speed, effi, cov = simulate(
            BO_FSS, prng, dist, P, N, h, params)
        x[i] = θ
        y[i] = time
    end
    push!(dataset_x, x[1:subsample]...)
    push!(dataset_y, y[1:subsample]...)
    push!(log, Dict("x"=>x, "y"=>y, "ir"=>minimum(dataset_y)))
    for i = 1:32
        θ, best_θ, best_y = MES(dataset_x, dataset_y, true, true)
        for t = 1:T
            params = Dict{Symbol, Any}(:param=>bo_fss_transform(θ));
            time, slow, speed, effi, cov = simulate(
                BO_FSS, prng, dist, P, N, h, params)
            x[t] = θ
            y[t] = time
        end
        x_sub = x[1:subsample]
        y_sub = y[1:subsample]
        push!(dataset_x, x_sub...)
        push!(dataset_y, y_sub...)

        for t = 1:T
            params = Dict{Symbol, Any}(:param=>bo_fss_transform(best_θ));
            time, slow, speed, effi, cov = simulate(
                BO_FSS, prng, dist, P, N, h, params)
            y[t] = time
        end
        push!(log, Dict("x"=>x_sub, "y"=>y_sub, "ir"=>minimum(dataset_y),
                        "sr"=>mean(y)))
    end
    return log
end


