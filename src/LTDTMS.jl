module LTDTMS

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
function get_site_stats_ml(T,E,K,Ethrprior,num_s_samples=100, verbose=true; kwargs...)
  thrdat = ThrData(T, E, K, Ethrprior)
  nsite = size(thrdat.EE,3)

  Z = zeros(Float64, (nsite,))
  Ethr = zeros(Float64, (nsite,))
  d = zeros(Float64, (nsite,3))
  s = zeros(Float64, (nsite,3))
  s_samples = zeros(Float64, (nsite, 3, num_s_samples))

  nthreads = Threads.nthreads()
    Threads.@threads :static for site in 1:nsite
        stats = get_site_stats(site, thrdat; kwargs...)
        Z[site] = stats[1]
        s[site,:] = stats[2]
        Ethr[site] = stats[3]
        d[site,:] = stats[4]
        s_samples[site,:,:] = hcat(stats[5][1:num_s_samples]...)
        if Threads.threadid() == 1 && site%10 == 0 && verbose
            println("site: $(site), approx. percentage: $(round(100*nthreads*site/nsite))") # TODO: use ProgressMeter or something similar
            flush(stdout)
        end
    end
  Dict(:Z => Z, :d => d, :Ethr => Ethr, :s => s)
end

end # module
