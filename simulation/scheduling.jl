
using JLD

include("../IBBO/IBBO.jl")
include("simulation.jl")

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
        w, μ, σ, H, best_θ, best_y = IBBO(dataset_x, dataset_y, true, true)
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
        push!(log, Dict("x"=>x, "y"=>y))
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

    if(sched <: BOSchedule)
        function bo_f(θ)
            locparams = Dict{Symbol, Any}(:param=>transform(θ));
            time, _, _, _, _ = simulate(
                BO_FSS, prng, gen, P, N, h, locparams)
            return time / (N / P)
        end
        log, best_θ = ibbo_bayesched(prng, bo_f, 32)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean]    = f
    params[:μ]       = mean(d)
    params[:σ]       = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params)
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
        log, best_θ = ibbo_bayesched(prng, bo_f, 32)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = mean(d)
    params[:σ]    = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params)
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
        log, best_θ = ibbo_bayesched(prng, bo_f, 32)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = mean(d)
    params[:σ]    = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params)
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
        log, best_θ = ibbo_bayesched(prng, bo_f, 32)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = mean(d)
    params[:σ]    = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params)
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
        w = begin
            if(i & (1 << 10) != 0)
                rand(prng, d) + a
            else
                rand(prng, d) + b
            end
        end
        return w
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
        log, best_θ = ibbo_bayesched(prng, bo_f, 32)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/4)
    return run_raw(sched, T, prng, gen, P, N, h, params)
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
        w = begin
            return (i*(b-a)/N + a) + rand(prng, d)
        end
        return w
    end
    function f(x)
        w = begin
            return (x*(b-a)/N + a)
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
        log, best_θ = ibbo_bayesched(prng, bo_f, 32)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/12)
    return run_raw(sched, T, prng, gen, P, N, h, params)
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
        w = begin
            return (b - i*(b-a)/N) + rand(prng, d)
        end
        return w
    end
    function f(x)
        w = begin
            return (b - x*(b-a)/N)
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
        log, best_θ = ibbo_bayesched(prng, bo_f, 32)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/12)
    return run_raw(sched, T, prng, gen, P, N, h, params)
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
        w = begin
            if(i & (1 << 7) != 0)
                rand(prng, d) + a
            else
                rand(prng, d) + b
            end
        end
        return w
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
        log, best_θ = ibbo_bayesched(prng, bo_f, 32)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/4)
    return run_raw(sched, T, prng, gen, P, N, h, params)
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
        w = begin
            return (i*(b-a)/N + a) + rand(prng, d)
        end
        return w
    end
    function f(x)
        w = begin
            return (x*(b-a)/N + a)
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
        log, best_θ = ibbo_bayesched(prng, bo_f, 32)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/12)
    return run_raw(sched, T, prng, gen, P, N, h, params)
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
        w = begin
            return (b - i*(b-a)/N) + rand(prng, d)
        end
        return w
    end
    function f(x)
        w = begin
            return (b - x*(b-a)/N)
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
        log, best_θ = ibbo_bayesched(prng, bo_f, 32)
        params[:param] = transform(best_θ)
    end
    params[:K_first] = N/(2*P)
    params[:K_last]  = 1
    params[:K_min]   = 1

    params[:mean] = f
    params[:μ]    = (b + a) / 2
    params[:σ]    = √(var(d) + (b - a)^2/12)
    return run_raw(sched, T, prng, gen, P, N, h, params)
end

function run_all(prng, path="")
    Random.seed!(prng)
    begin
        println("------- BO_FSS -------")
        d      = Dict()
        sched  = BO_FSS
        transf = bo_fss_transform
        d["b1"]  = vcat([bench1(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:2]...)
        d["b2"]  = vcat([bench2(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:2]...)
        d["b3"]  = vcat([bench3(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:2]...)
        d["b4"]  = vcat([bench4(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:2]...)
        d["b5"]  = vcat([bench5(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:2]...)
        d["b6"]  = vcat([bench6(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:2]...)
        d["b7"]  = vcat([bench7(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:2]...)
        d["b8"]  = vcat([bench8(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:2]...)
        d["b9"]  = vcat([bench9(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:2]...)
        d["b10"] = vcat([bench10(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:2]...)
        save(joinpath(path, "BO_FSS.jld"), d)
    end

    begin
        println("------- FSS -------")
        d     = Dict()
        sched = FSS

        d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)
        save(joinpath(path, "FSS.jld"), d)
    end

    begin
        println("------- FAC2 -------")
        d     = Dict()
        sched = FAC2

        d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)
        save(joinpath(path, "FAC2.jld"), d)
    end

    begin
        println("------- CSS -------")
        d     = Dict()
        sched = CSS

        d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)
        save(joinpath(path, "CSS.jld"), d)
    end

    begin
        d     = Dict()
        sched = TSS
        d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)
        save(joinpath(path, "TSS.jld"), d)
    end

    begin
        println("------- TAPER -------")
        d     = Dict()
        sched = TAPER
        d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)
        save(joinpath(path, "TAPER.jld"), d)
    end

    begin
        println("------- TAPER3 -------")
        d     = Dict()
        sched = TAPER3
        d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)
        save(joinpath(path, "TAPER3.jld"), d)
    end

    begin
        println("------- AFAC -------")
        d     = Dict()
        sched = AFAC
        d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)
        save(joinpath(path, "AFAC.jld"), d)
    end

    begin
        println("------- HSS -------")
        d     = Dict()
        sched = HSS
        d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)
        save(joinpath(path, "HSS.jld"), d)
    end

    begin
        println("------- BinLPT -------")
        d     = Dict()
        sched = BinLPT
        d["b1"]  = bench1(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b2"]  = bench2(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b3"]  = bench3(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b4"]  = bench4(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b5"]  = bench5(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b6"]  = bench6(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b7"]  = bench7(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b8"]  = bench8(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b9"]  = bench9(prng,  sched, Dict{Symbol, Any}(), transform=nothing)
        d["b10"] = bench10(prng, sched, Dict{Symbol, Any}(), transform=nothing)
        save(joinpath(path, "BinLPT.jld"), d)
    end

    begin
        println("------- BO_FAC -------")
        d      = Dict()
        sched  = BO_FAC
        transf = bo_fac_transform
        d["b1"]  = vcat([bench1(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b2"]  = vcat([bench2(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b3"]  = vcat([bench3(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b4"]  = vcat([bench4(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b5"]  = vcat([bench5(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b6"]  = vcat([bench6(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b7"]  = vcat([bench7(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b8"]  = vcat([bench8(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b9"]  = vcat([bench9(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b10"] = vcat([bench10(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        save(joinpath(path, "BO_FAC.jld"), d)
    end

    begin
        println("------- BO_TSS -------")
        d      = Dict()
        sched  = BO_TSS
        transf = bo_tss_transform
        d["b1"]  = vcat([bench1(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b2"]  = vcat([bench2(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b3"]  = vcat([bench3(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b4"]  = vcat([bench4(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b5"]  = vcat([bench5(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b6"]  = vcat([bench6(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b7"]  = vcat([bench7(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b8"]  = vcat([bench8(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b9"]  = vcat([bench9(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b10"] = vcat([bench10(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        save(joinpath(path, "BO_TSS.jld"), d)
    end

    begin
        println("------- BO_CSS -------")
        d      = Dict()
        sched  = BO_CSS
        transf = bo_css_transform
        d["b1"]  = vcat([bench1(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b2"]  = vcat([bench2(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b3"]  = vcat([bench3(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b4"]  = vcat([bench4(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b5"]  = vcat([bench5(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b6"]  = vcat([bench6(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b7"]  = vcat([bench7(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b8"]  = vcat([bench8(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b9"]  = vcat([bench9(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b10"] = vcat([bench10(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        save(joinpath(path, "BO_CSS.jld"), d)
    end

    begin
        println("------- BO_TAPER -------")
        d      = Dict()
        sched  = BO_TAPER
        transf = bo_taper_transform
        d["b1"]  = vcat([bench1(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b2"]  = vcat([bench2(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b3"]  = vcat([bench3(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b4"]  = vcat([bench4(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b5"]  = vcat([bench5(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b6"]  = vcat([bench6(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b7"]  = vcat([bench7(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b8"]  = vcat([bench8(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b9"]  = vcat([bench9(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        d["b10"] = vcat([bench10(prng,  sched, Dict{Symbol, Any}(), transform=transf) for i = 1:8]...)
        save(joinpath(path, "BO_TAPER.jld"), d)
    end
end
