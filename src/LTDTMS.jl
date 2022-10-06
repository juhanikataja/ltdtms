
module LTDTMS
#= include("lpdfs.jl") =#
#= include("treeint.jl") =#

#= using .Lpdfs: ThrData =#
include("lpdfs.jl")
include("treeint.jl")
include("posterior.jl")

import .Lpdfs: lpost, ThrData, lpost_simple

export get_site_stats

## The computing functions
function makeThrData(indat::Dict) 

  EE = indat[:EE]
  K = indat[:K]
  minThr = indat[:minThr]
  h = indat[:resolution]
  MT = [t for t in indat[:MT]][:]

  return ThrData(MT, EE, K, 1/minThr, h)
end 


end # module
