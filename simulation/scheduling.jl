
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
    T = 128
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
        log, best_θ = ibbo_bayesched(prng, f, 32)
        params[:param] = transform(best_θ)
    end
    params[:mean] = f
    params[:μ]    = mean(d)
    params[:σ]    = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params)
end

function bench2(prng, sched, params)
    T = 128
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
    params[:mean] = f
    params[:μ]    = mean(d)
    params[:σ]    = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params)
end

function bench3(prng, sched, params)
    T = 128
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
    params[:mean] = f
    params[:μ]    = mean(d)
    params[:σ]    = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params)
end

function bench4(prng, sched, params)
    T = 128
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
    params[:mean] = f
    params[:μ]    = mean(d)
    params[:σ]    = std(d)
    return run_raw(sched, T, prng, gen, P, N, h, params)
end

function run_all(prng, )
    
end
