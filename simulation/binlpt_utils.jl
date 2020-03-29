
function offset_cumsum(arr)
    sum = zeros(eltype(arr), length(arr))
    for i = 2:length(arr)
        sum[i] = sum[i - 1] + arr[i - 1];
    end
    return (sum);
end

function compute_chunksizes(tasks, ntasks, nchunks)
    chunksizes  = zeros(Int64, nchunks)
    workload    = offset_cumsum(tasks)
    chunkweight = (workload[end] + tasks[end])/nchunks;
    k = 1
    i = 1
    while (i <= ntasks)
        j = i + 1
        while (j <= ntasks)
            if (workload[j] - workload[i] >= chunkweight)
                break
            end
            j += 1
        end
        chunksizes[k] = j - i
        i  = j
        k += 1
    end
    return chunksizes;
end

function compute_chunkweights(tasks, ntasks, chunksizes, nchunks)
    chunks = zeros(Float64, nchunks)

    # Compute chunks.
    k = 1
    for i = 1:nchunks
        for j = 1:chunksizes[i]
            chunks[i] += tasks[k]
            k += 1
        end
    end
    return chunks;
end

function binlpt_balance(tasks, P, K_max)
    N = length(tasks)
    load = zeros(P)
    taskmap = zeros(UInt16, N) 

    chunksizes = compute_chunksizes(tasks, N, K_max)
    chunks     = compute_chunkweights(tasks, N, chunksizes, K_max)
    chunkoff   = offset_cumsum(chunksizes)

    # Sort tasks.
    sortmap = sortperm(chunks)

    for i = K_max:-1:1
        if (chunks[sortmap[i]] == 0)
            continue
        end
        tid = 1
        for j = 2:P
            if (load[j] < load[tid])
                tid = j
            end
        end

        for j = 1:chunksizes[sortmap[i]]
            taskmap[chunkoff[sortmap[i]] + j] = tid
        end
        load[tid] += chunks[sortmap[i]]
    end
    return taskmap;
end

function binlpt_balance(f, N, P, K_max)
    tasks   = f.(1:N)
    load    = zeros(P)
    taskmap = zeros(UInt16, N) 

    chunksizes = compute_chunksizes(tasks, N, K_max)
    chunks     = compute_chunkweights(tasks, N, chunksizes, K_max)
    chunkoff   = offset_cumsum(chunksizes)

    # Sort tasks.
    sortmap = sortperm(chunks)

    for i = K_max:-1:1
        if (chunks[sortmap[i]] == 0)
            continue
        end
        tid = 1
        for j = 2:P
            if (load[j] < load[tid])
                tid = j
            end
        end

        for j = 1:chunksizes[sortmap[i]]
            taskmap[chunkoff[sortmap[i]] + j] = tid
        end
        load[tid] += chunks[sortmap[i]]
    end
    return taskmap;
end
