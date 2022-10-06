
module LTDTMS
#= include("lpdfs.jl") =#
#= include("treeint.jl") =#

#= using .Lpdfs: ThrData =#
include("lpdfs.jl")
include("treeint.jl")
include("posterior.jl")

import .Lpdfs: lpost, ThrData, lpost_simple

## The computing functions
function makeThrData(indat::Dict) 

  EE = indat[:EE]
  K = indat[:K]
  minThr = indat[:minThr]
  h = indat[:resolution]
  MT = [t for t in indat[:MT]][:]

  i_eq = findall(x-> (x>0 && x <1), MT)

  #= MT = MT[i_eq] =#
  #= EE = EE[i_eq, :, :] =#

  # TODO: clean this up

  return ThrData(MT, EE, K, 1/minThr, h)
end 


end # module
