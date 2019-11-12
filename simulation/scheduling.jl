
include("../IBBO/IBBO.jl")	
include("simulation.jl")	

using JLD	
using .IBBO

function ibbo_bayesched(prng, f, T)
    log = Dict[]
    x = zeros(Float64, T)
    y = zeros(Float64, T)

    dataset_x = Float64[]
    dataset_y = Float64[]
    
    init_d = Uniform(0.0, 1.0)
    for i = 1:T
        θ    = rand(prng, init_d)
        time = f(θ)
        x[i] = θ
        y[i] = time
    end

    push!(dataset_x, x...)
    push!(dataset_y, y...)
    push!(log, Dict("x"=>x, "y"=>y))

    best_θ = 0
    for i = 1:8
        println("--------- iteration $i -----------")
        println(" dataset size = $(length(dataset_x))")
        w, μ, σ, H, αx, αy, gpx, gpμ, gpσ, samples, best_θ, best_y =
            ibbo_log(dataset_x, -dataset_y, true, true)

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
            time = f(θ)
            x[t] = θ
            y[t] = time
        end
        push!(dataset_x, x...)
        push!(dataset_y, y...)
        println("length: $(length(dataset_x)), $(length(dataset_y))")
        d = Dict()
        d["x"] = x
        d["y"] = y
        d["w"] = w
        d["μ"] = μ
        d["σ"] = σ
        d["H"] = H
        d["αx"] = αx
        d["αy"] = αy
        d["gpx"] = gpx
        d["gpμ"] = gpμ
        d["gpσ"] = gpσ
        d["samples"] = samples
        d["best_θ"]  = best_θ
        d["best_y"]  = best_y
        push!(log, d)
        GC.gc()
    end
    return log, best_θ
end

function bench1(prng, sched, params; transform=nothing)
    # Normal distribution
    T = 512
    N = 2^20
    P = 128
    h = 0.1
    d = TruncatedNormal(1.0, 0.1, 0.0, Inf)
    function gen(prng, i, N)
        return rand(prng, d)
    end
    function f(x)
        return 1
    end

    log = nothing
    if(sched <: BOSchedule)
        function bo_f(θ)
            locparams = Dict{Symbol, Any}(:param=>transform(θ));
            time, _, _, _, _ = simulate(
                BO_FSS, prng, gen, P, N, h, locparams)
            return time / (N / P)
        end
        log, best_θ = ibbo_bayesched(prng, bo_f, 64)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean]    = f
    params[:μ]       = mean(d)
    params[:σ]       = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params), log
end

function bench2(prng, sched, params; transform=nothing)
    # Exponential distribution

    T = 512
    N = 2^20
    P = 128
    h = 0.1
    d = Exponential(1.0)
    function gen(prng, i, N)
        return rand(prng, d)
    end
    function f(x)
        return 1
    end
    if(sched <: BOSchedule)
        function bo_f(θ)
            locparams = Dict{Symbol, Any}(:param=>transform(θ));
            time, _, _, _, _ = simulate(
                BO_FSS, prng, gen, P, N, h, locparams)
            return time / (N / P)
        end
        log, best_θ = ibbo_bayesched(prng, bo_f, 64)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = mean(d)
    params[:σ]    = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params), log
end

function bench3(prng, sched, params; transform=nothing)
    # Normal distribution Low iteration count

    T = 512
    N = 2^10
    P = 128
    h = 0.1
    d = TruncatedNormal(1.0, 0.1, 0.0, Inf)
    function gen(prng, i, N)
        return rand(prng, d)
    end
    function f(x)
        return 1
    end
    if(sched <: BOSchedule)
        function bo_f(θ)
            locparams = Dict{Symbol, Any}(:param=>transform(θ));
            time, _, _, _, _ = simulate(
                BO_FSS, prng, gen, P, N, h, locparams)
            return time / (N / P)
        end
        log, best_θ = ibbo_bayesched(prng, bo_f, 64)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = mean(d)
    params[:σ]    = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params), log
end

function bench4(prng, sched, params; transform=nothing)
    # Exponential distribution low iteration count

    T = 512
    N = 2^10
    P = 128
    h = 0.1
    d = Exponential(1.0)
    function gen(prng, i, N)
        return rand(prng, d)
    end
    function f(x)
        return 1
    end
    if(sched <: BOSchedule)
        function bo_f(θ)
            locparams = Dict{Symbol, Any}(:param=>transform(θ));
            time, _, _, _, _ = simulate(
                BO_FSS, prng, gen, P, N, h, locparams)
            return time / (N / P)
        end
        log, best_θ = ibbo_bayesched(prng, bo_f, 64)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = mean(d)
    params[:σ]    = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params), log
end

function bench5(prng, sched, params; transform=nothing)
    # Biased distribution, Square 
    T = 512
    N = 2^20
    P = 128
    h = 0.1
    σ = 0.1
    a = 1
    b = 5
    d = TruncatedNormal(0.0, σ, 0.0, Inf)
    function gen(prng, i, N)
        if(i & (1 << 10) != 0)
            return rand(prng, d) + a
        else
            return rand(prng, d) + b
        end
    end
    function f(x)
        w = begin
            if(x & (1 << 10) != 0)
                a
            else
                b
            end
        end
        return w
    end
    if(sched <: BOSchedule)
        function bo_f(θ)
            locparams = Dict{Symbol, Any}(:param=>transform(θ));
            time, _, _, _, _ = simulate(
                BO_FSS, prng, gen, P, N, h, locparams)
            return time / (N / P)
        end
        log, best_θ = ibbo_bayesched(prng, bo_f, 64)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/4)
    return run_raw(sched, T, prng, gen, P, N, h, params), log
end

function bench6(prng, sched, params; transform=nothing)
    # Biased distribution, Increasing
    T = 512
    N = 2^20
    P = 128
    h = 0.1
    σ = 0.1
    a = 1
    b = 5
    d = TruncatedNormal(0.0, σ, 0.0, Inf)
    function gen(prng, i, N)
        return (i*(b-a)/N + a) + rand(prng, d)
    end
    function f(x)
        return (x*(b-a)/N + a)
    end
    if(sched <: BOSchedule)
        function bo_f(θ)
            locparams = Dict{Symbol, Any}(:param=>transform(θ));
            time, _, _, _, _ = simulate(
                BO_FSS, prng, gen, P, N, h, locparams)
            return time / (N / P)
        end
        log, best_θ = ibbo_bayesched(prng, bo_f, 64)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/12)
    return run_raw(sched, T, prng, gen, P, N, h, params), log
end

function bench7(prng, sched, params; transform=nothing)
    # Biased distribution, Decreasing
    T = 512
    N = 2^20
    P = 128
    h = 0.1
    σ = 0.1
    a = 1
    b = 5
    d = TruncatedNormal(0.0, σ, 0.0, Inf)
    function gen(prng, i, N)
        return (b - i*(b-a)/N) + rand(prng, d)
    end
    function f(x)
        return b - x*(b-a)/N
    end
    if(sched <: BOSchedule)
        function bo_f(θ)
            locparams = Dict{Symbol, Any}(:param=>transform(θ));
            time, _, _, _, _ = simulate(
                BO_FSS, prng, gen, P, N, h, locparams)
            return time / (N / P)
        end
        log, best_θ = ibbo_bayesched(prng, bo_f, 64)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/12)
    return run_raw(sched, T, prng, gen, P, N, h, params), log
end

function bench8(prng, sched, params; transform=nothing)
    # Biased distribution, Square, low iteration count
    T = 512
    N = 2^10
    P = 128
    h = 0.1
    σ = 0.1
    a = 1
    b = 5
    d = TruncatedNormal(0.0, σ, 0.0, Inf)
    function gen(prng, i, N)
        if(i & (1 << 7) != 0)
            return rand(prng, d) + a
        else
            return rand(prng, d) + b
        end
    end
    function f(x)
        if(x & (1 << 10) != 0)
            return a
        else
            return b
        end
    end
    if(sched <: BOSchedule)
        function bo_f(θ)
            locparams = Dict{Symbol, Any}(:param=>transform(θ));
            time, _, _, _, _ = simulate(
                BO_FSS, prng, gen, P, N, h, locparams)
            return time / (N / P)
        end
        log, best_θ = ibbo_bayesched(prng, bo_f, 64)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/4)
    return run_raw(sched, T, prng, gen, P, N, h, params), log
end

function bench9(prng, sched, params; transform=nothing)
    # Biased distribution, Increasing, Low iteration

    T = 512
    N = 2^10
    P = 128
    h = 0.1
    σ = 0.1
    a = 1
    b = 5
    d = TruncatedNormal(0.0, σ, 0.0, Inf)
    function gen(prng, i, N)
        return (i*(b-a)/N + a) + rand(prng, d)
    end
    function f(x)
        return (x*(b-a)/N + a)
    end
    if(sched <: BOSchedule)
        function bo_f(θ)
            locparams = Dict{Symbol, Any}(:param=>transform(θ));
            time, _, _, _, _ = simulate(
                BO_FSS, prng, gen, P, N, h, locparams)
            return time / (N / P)
        end
        log, best_θ = ibbo_bayesched(prng, bo_f, 64)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/12)
    return run_raw(sched, T, prng, gen, P, N, h, params), log
end

function bench10(prng, sched, params; transform=nothing)
    # Biased distribution, Decreasing, Low iteration

    T = 512
    N = 2^10
    P = 128
    h = 0.1
    σ = 0.1
    a = 1
    b = 5
    d = TruncatedNormal(0.0, σ, 0.0, Inf)
    function gen(prng, i, N)
        return (b - i*(b-a)/N) + rand(prng, d)
    end
    function f(x)
        return (b - x*(b-a)/N)
    end
    if(sched <: BOSchedule)
        function bo_f(θ)
            locparams = Dict{Symbol, Any}(:param=>transform(θ));
            time, _, _, _, _ = simulate(
                BO_FSS, prng, gen, P, N, h, locparams)
            return time / (N / P)
        end
        log, best_θ = ibbo_bayesched(prng, bo_f, 64)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/12)
    return run_raw(sched, T, prng, gen, P, N, h, params), log
end

function run_all(prng, path="")
    Random.seed!(prng)
    # begin
    #     println("------- FSS -------")
    #     d     = Dict()
    #     sched = FSS

    #     d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     save(joinpath(path, "FSS.jld"), d)
    # end

    # begin
    #     println("------- FAC2 -------")
    #     d     = Dict()
    #     sched = FAC2

    #     d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     save(joinpath(path, "FAC2.jld"), d)
    # end

    # begin
    #     println("------- CSS -------")
    #     d     = Dict()
    #     sched = CSS

    #     d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     save(joinpath(path, "CSS.jld"), d)
    # end

    # begin
    #     d     = Dict()
    #     sched = TSS
    #     d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     save(joinpath(path, "TSS.jld"), d)
    # end

    # begin
    #     println("------- TAPER -------")
    #     d     = Dict()
    #     sched = TAPER
    #     d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(:α=>1.3), transform=nothing)[1]
    #     d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(:α=>1.3), transform=nothing)[1]
    #     d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(:α=>1.3), transform=nothing)[1]
    #     d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(:α=>1.3), transform=nothing)[1]
    #     d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(:α=>1.3), transform=nothing)[1]
    #     d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(:α=>1.3), transform=nothing)[1]
    #     d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(:α=>1.3), transform=nothing)[1]
    #     d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(:α=>1.3), transform=nothing)[1]
    #     d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(:α=>1.3), transform=nothing)[1]
    #     d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(:α=>1.3), transform=nothing)[1]
    #     save(joinpath(path, "TAPER.jld"), d)
    # end

    # begin
    #     println("------- TAPER3 -------")
    #     d     = Dict()
    #     sched = TAPER3
    #     d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     save(joinpath(path, "TAPER3.jld"), d)
    # end

    # begin
    #     println("------- AFAC -------")
    #     d     = Dict()
    #     sched = AFAC
    #     d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     save(joinpath(path, "AFAC.jld"), d)
    # end

    # begin
    #     println("------- HSS -------")
    #     d     = Dict()
    #     sched = HSS
    #     d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     save(joinpath(path, "HSS.jld"), d)
    # end

    # begin
    #     println("------- BinLPT -------")
    #     d     = Dict()
    #     sched = BinLPT
    #     d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)[1]
    #     save(joinpath(path, "BinLPT.jld"), d)
    # end

    begin
        println("------- BO_FSS -------")
        d      = Dict()
        sched  = BO_FSS
        transf = bo_fss_transform
        b1  = [bench1(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b2  = [bench2(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b3  = [bench3(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b4  = [bench4(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b5  = [bench5(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b6  = [bench6(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b7  = [bench7(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b8  = [bench8(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b9  = [bench9(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b10 = [bench10(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        d["b1"]      = vcat([i[1] for i in b1]...)
        d["b1_log"]  = [i[2] for i in b1]
        d["b2"]      = vcat([i[1] for i in b2]...)
        d["b2_log"]  = [i[2] for i in b2]
        d["b3"]      = vcat([i[1] for i in b3]...)
        d["b3_log"]  = [i[2] for i in b3]
        d["b4"]      = vcat([i[1] for i in b4]...)
        d["b4_log"]  = [i[2] for i in b4]
        d["b4"]      = vcat([i[1] for i in b4]...)
        d["b4_log"]  = [i[2] for i in b4]
        d["b5"]      = vcat([i[1] for i in b5]...)
        d["b5_log"]  = [i[2] for i in b5]
        d["b6"]      = vcat([i[1] for i in b6]...)
        d["b6_log"]  = [i[2] for i in b6]
        d["b7"]      = vcat([i[1] for i in b7]...)
        d["b7_log"]  = [i[2] for i in b7]
        d["b8"]      = vcat([i[1] for i in b8]...)
        d["b8_log"]  = [i[2] for i in b8]
        d["b9"]      = vcat([i[1] for i in b9]...)
        d["b9_log"]  = [i[2] for i in b9]
        d["b10"]     = vcat([i[1] for i in b10]...)
        d["b10_log"] = [i[2] for i in b10]
        save(joinpath(path, "BO_FSS.jld"), d)
    end


    begin
        println("------- BO_FAC -------")
        d      = Dict()
        sched  = BO_FAC
        transf = bo_fac_transform
        b1  = [bench1(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b2  = [bench2(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b3  = [bench3(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b4  = [bench4(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b5  = [bench5(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b6  = [bench6(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b7  = [bench7(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b8  = [bench8(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b9  = [bench9(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b10 = [bench10(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        d["b1"]      = vcat([i[1] for i in b1]...)
        d["b1_log"]  = [i[2] for i in b1]
        d["b2"]      = vcat([i[1] for i in b2]...)
        d["b2_log"]  = [i[2] for i in b2]
        d["b3"]      = vcat([i[1] for i in b3]...)
        d["b3_log"]  = [i[2] for i in b3]
        d["b4"]      = vcat([i[1] for i in b4]...)
        d["b4_log"]  = [i[2] for i in b4]
        d["b4"]      = vcat([i[1] for i in b4]...)
        d["b4_log"]  = [i[2] for i in b4]
        d["b5"]      = vcat([i[1] for i in b5]...)
        d["b5_log"]  = [i[2] for i in b5]
        d["b6"]      = vcat([i[1] for i in b6]...)
        d["b6_log"]  = [i[2] for i in b6]
        d["b7"]      = vcat([i[1] for i in b7]...)
        d["b7_log"]  = [i[2] for i in b7]
        d["b8"]      = vcat([i[1] for i in b8]...)
        d["b8_log"]  = [i[2] for i in b8]
        d["b9"]      = vcat([i[1] for i in b9]...)
        d["b9_log"]  = [i[2] for i in b9]
        d["b10"]     = vcat([i[1] for i in b10]...)
        d["b10_log"] = [i[2] for i in b10]
        save(joinpath(path, "BO_FAC.jld"), d)
    end

    begin
        println("------- BO_TSS -------")
        d      = Dict()
        sched  = BO_TSS
        transf = x->x #bo_tss_transform
        b1  = [bench1(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b2  = [bench2(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b3  = [bench3(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b4  = [bench4(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b5  = [bench5(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b6  = [bench6(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b7  = [bench7(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b8  = [bench8(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b9  = [bench9(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b10 = [bench10(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        d["b1"]      = vcat([i[1] for i in b1]...)
        d["b1_log"]  = [i[2] for i in b1]
        d["b2"]      = vcat([i[1] for i in b2]...)
        d["b2_log"]  = [i[2] for i in b2]
        d["b3"]      = vcat([i[1] for i in b3]...)
        d["b3_log"]  = [i[2] for i in b3]
        d["b4"]      = vcat([i[1] for i in b4]...)
        d["b4_log"]  = [i[2] for i in b4]
        d["b4"]      = vcat([i[1] for i in b4]...)
        d["b4_log"]  = [i[2] for i in b4]
        d["b5"]      = vcat([i[1] for i in b5]...)
        d["b5_log"]  = [i[2] for i in b5]
        d["b6"]      = vcat([i[1] for i in b6]...)
        d["b6_log"]  = [i[2] for i in b6]
        d["b7"]      = vcat([i[1] for i in b7]...)
        d["b7_log"]  = [i[2] for i in b7]
        d["b8"]      = vcat([i[1] for i in b8]...)
        d["b8_log"]  = [i[2] for i in b8]
        d["b9"]      = vcat([i[1] for i in b9]...)
        d["b9_log"]  = [i[2] for i in b9]
        d["b10"]     = vcat([i[1] for i in b10]...)
        d["b10_log"] = [i[2] for i in b10]
        save(joinpath(path, "BO_TSS.jld"), d)
    end

    begin
        println("------- BO_CSS -------")
        d      = Dict()
        sched  = BO_CSS
        transf = bo_css_transform
        b1  = [bench1(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b2  = [bench2(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b3  = [bench3(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b4  = [bench4(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b5  = [bench5(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b6  = [bench6(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b7  = [bench7(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b8  = [bench8(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b9  = [bench9(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b10 = [bench10(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        d["b1"]      = vcat([i[1] for i in b1]...)
        d["b1_log"]  = [i[2] for i in b1]
        d["b2"]      = vcat([i[1] for i in b2]...)
        d["b2_log"]  = [i[2] for i in b2]
        d["b3"]      = vcat([i[1] for i in b3]...)
        d["b3_log"]  = [i[2] for i in b3]
        d["b4"]      = vcat([i[1] for i in b4]...)
        d["b4_log"]  = [i[2] for i in b4]
        d["b4"]      = vcat([i[1] for i in b4]...)
        d["b4_log"]  = [i[2] for i in b4]
        d["b5"]      = vcat([i[1] for i in b5]...)
        d["b5_log"]  = [i[2] for i in b5]
        d["b6"]      = vcat([i[1] for i in b6]...)
        d["b6_log"]  = [i[2] for i in b6]
        d["b7"]      = vcat([i[1] for i in b7]...)
        d["b7_log"]  = [i[2] for i in b7]
        d["b8"]      = vcat([i[1] for i in b8]...)
        d["b8_log"]  = [i[2] for i in b8]
        d["b9"]      = vcat([i[1] for i in b9]...)
        d["b9_log"]  = [i[2] for i in b9]
        d["b10"]     = vcat([i[1] for i in b10]...)
        d["b10_log"] = [i[2] for i in b10]
        save(joinpath(path, "BO_CSS.jld"), d)
    end

    begin
        println("------- BO_TAPER -------")
        d      = Dict()
        sched  = BO_TAPER
        transf = bo_taper_transform
        b1  = [bench1(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b2  = [bench2(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b3  = [bench3(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b4  = [bench4(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b5  = [bench5(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b6  = [bench6(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b7  = [bench7(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b8  = [bench8(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b9  = [bench9(prng,   sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        b10 = [bench10(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:32]
        d["b1"]      = vcat([i[1] for i in b1]...)
        d["b1_log"]  = [i[2] for i in b1]
        d["b2"]      = vcat([i[1] for i in b2]...)
        d["b2_log"]  = [i[2] for i in b2]
        d["b3"]      = vcat([i[1] for i in b3]...)
        d["b3_log"]  = [i[2] for i in b3]
        d["b4"]      = vcat([i[1] for i in b4]...)
        d["b4_log"]  = [i[2] for i in b4]
        d["b4"]      = vcat([i[1] for i in b4]...)
        d["b4_log"]  = [i[2] for i in b4]
        d["b5"]      = vcat([i[1] for i in b5]...)
        d["b5_log"]  = [i[2] for i in b5]
        d["b6"]      = vcat([i[1] for i in b6]...)
        d["b6_log"]  = [i[2] for i in b6]
        d["b7"]      = vcat([i[1] for i in b7]...)
        d["b7_log"]  = [i[2] for i in b7]
        d["b8"]      = vcat([i[1] for i in b8]...)
        d["b8_log"]  = [i[2] for i in b8]
        d["b9"]      = vcat([i[1] for i in b9]...)
        d["b9_log"]  = [i[2] for i in b9]
        d["b10"]     = vcat([i[1] for i in b10]...)
        d["b10_log"] = [i[2] for i in b10]
        save(joinpath(path, "BO_TAPER.jld"), d)
    end
end
