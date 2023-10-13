module LTDTMS
using ProgressMeter

include("lpdfs.jl")
include("treeint.jl")
include("posterior.jl")

import .Lpdfs: lpost, ThrData, lpost_simple

export get_site_stats, makeThrData, get_site_stats_ml, ThrData, comparisonMLH

## mangle a Dictionary to ThrData
function makeThrData(indat::Dict) 

  EE = indat[:EE]
  K = indat[:K]
  minThr = indat[:minThr]
  h = indat[:resolution]
  MT = [t for t in indat[:MT]][:]

  return ThrData(MT, EE, K, minThr)
end 

"""
    get_site_stats_ml(T,E,K,Ethrprior; kwargs...)

Wrapper for `get_site_stats(site, dat; kwargs...)` that plays well with
MATDaemon.jl returning the results in Dictionary format with keys `(:Z, :d, :Ethr, :s)`.

Uses threads. For keyword arguments, see [`get_site_stats`](@ref).

"""
function get_site_stats_ml(T,E,K,Ethrprior,num_s_samples=100, verbose=true, save_samples=false; kwargs...)
    thrdat = ThrData(T, E, K, Ethrprior)
    nsite = size(thrdat.EE,3)

    Z = zeros(Float64, (nsite,))
    Ethr = zeros(Float64, (nsite,))
    d = zeros(Float64, (nsite,3))
    s = zeros(Float64, (nsite,3))

    num_s_samples = ifelse(haskey(kwargs,:n_samples), kwargs[:n_samples], num_s_samples)
    s_samples = zeros(Float64, (nsite, 3, num_s_samples))
    progress_meter = Progress(nsite; output=stdout)

    nthreads = Threads.nthreads()

    Threads.@threads for site in 1:nsite
        stats = get_site_stats(site, thrdat; kwargs...)
        Z[site] = stats[1]
        s[site,:] = stats[2]
        Ethr[site] = stats[3]
        d[site,:] = stats[4]
        if save_samples
            s_samples[site,:,:] = hcat(stats[5][1:num_s_samples]...)
        end
        if verbose
            next!(progress_meter)
            flush(stdout)
        end
    end
    finish!(progress_meter)

    if save_samples
        Dict(:Z => Z, :d => d, :Ethr => Ethr, :s => s, :s_samples=>s_samples)
    else
        Dict(:Z => Z, :d => d, :Ethr => Ethr, :s => s)
    end
end

end # module
