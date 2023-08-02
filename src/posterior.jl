using AdvancedHMC
using LinearAlgebra
using LogDensityProblems
import ForwardDiff
using Optim
import ForwardDiff: Dual
import .treeint
#= import Logging =#

import .Lpdfs: lpost, ThrData, lpost_simple


function get_samples(dat::ThrData, n,lambda;n_samples=2000, progress=false, n_adapts=1000) 
  D = 3;
  (initial_th,initial_lpdf_val) = begin
    map = optimize( s->-lpost(s, lambda, dat, n), 1.0e-3.*[1,1,1]; autodiff=:forward)
    (map.minimizer, -map.minimum)
  end

  metric=DenseEuclideanMetric(D)

  hamiltonian = Hamiltonian(metric, LogTargetDensity(D, s->lpost(s, lambda, dat, n)), ForwardDiff)

  initial_e = find_good_stepsize(hamiltonian, initial_th)
  integrator = Leapfrog(initial_e)

  proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
  adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric),
                         StepSizeAdaptor(0.8, integrator));
  samples, stats = sample(
    hamiltonian, proposal,
    initial_th, n_samples, adaptor, n_adapts;
    progress=progress, verbose=false, 
    drop_warmup=true)

  return samples, stats, initial_lpdf_val, initial_th
end 

function get_samples(dat::ThrData, lpdf::Function, n; n_samples=2000, progress=false, n_adapts=1000)
  D=3;
  (initial_th,initial_lpdf_val) = begin
    map = optimize( s->-lpdf(s, dat, n), 1.0e-3.*[1,1,1]; autodiff=:forward)
    (map.minimizer, -map.minimum)
  end

  metric=DenseEuclideanMetric(D)

  hamiltonian = Hamiltonian(metric, LogTargetDensity(D, s->lpdf(s, dat, n)), ForwardDiff)

  initial_e = find_good_eps(hamiltonian, initial_th)
  integrator = Leapfrog(initial_e)

  proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
  adaptor = StanHMCAdaptor(Preconditioner(metric),
                         NesterovDualAveraging(0.8, integrator));
  samples, stats = sample(
    hamiltonian, proposal,
    initial_th, n_samples, adaptor, n_adapts;
    progress=progress, verbose=false, 
    drop_warmup=true)

  return samples, stats, initial_lpdf_val, initial_th
end 

# stereo maps and dets {{{

function det_st_map(s,α)
  return 4*α^3*s[3]^2 / ((1+s[1]^2+s[2]^2)^2)
end

function st_map(s,Q,α)
  P = s[1]^2+s[2]^2
  S0 = Q*[(P-1), 2*s[1], 2*s[2]]
  return S0.*(α*s[3]/(P+1))
end

function stereo_post(s, T, E, K, Ethrp, Q, α)
  s_mapped = st_map(s,Q,α)
  lpost_simple(s_mapped, T, E, K,  1/Ethrp) + log(det_st_map(s,α))
end

function stereo_post(s, α, dat::ThrData, site, Q)
  s_mapped = st_map(s,Q,α)
  @inbounds lpost_simple(s_mapped, dat.MT, dat.EE[:,:,site], dat.K, dat.smax) + log(det_st_map(s,α))
  #= lpost_simple(st_map(s,Q,α), dat, site) + log(det_st_map(s,α)) =#
end
# End of stereo maps and dets (don't export these) }}}


struct LogTargetDensity
    dim::Int
    logpi::Function
end


LogDensityProblems.logdensity(p::LogTargetDensity, θ) = p.logpi(θ)
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

function get_samples_nuts(logπ, initial_θ, D::Int64; 
    n_samples=1000, n_adapts=200, progress=false)
    
    ℓπ = LogTargetDensity(D,logπ)

    metric = DenseEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
    integrator = Leapfrog(find_good_stepsize(hamiltonian, initial_θ))
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    samples, stats = sample(hamiltonian, 
                            proposal,
                            initial_θ, 
                            n_samples+n_adapts, 
                            StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator)),
                            n_adapts;
                            progress=progress, verbose=false, 
                            drop_warmup=true)
    return samples, stats

end

"""
    get_site_stats(T, E, K, Ethrprior; kwargs...)

Given threshold vector `T`, field data array `E` (``N_{thr} \\times 3 \\times N_{site}``),
hyperparameter `K`, prior threshold `Ethrprior`, returns LTD statistics at each
site. 
"""
function get_site_stats(T, E, K, Ethrprior; kwargs...)
    get_site_stats(1:size(E,3), Lpdfs.ThrData(T,E,K,Ethrprior); kwargs...)
end

"""
    get_site_stats(sites::Union{Int64, Array{Int64,1}, UnitRange{Int64}}, dat::ThrData; n_samples = 1000, n_adapts=200, progress=false, treedepth=5) 

Given `sites` and threshold data `dat`, returns LTD statistics as an array of n-tuples consisting of data
``(Z, \\mathbf E [s], \\mathbf E [E_{thr}], \\mathbf E[d])``, where ``Z`` is
the normalizing coefficient at that site.


# Keyword arguments

* `n_samples` number of HMC samples. 
* `n_adapts` number of adaptation samples
* `progress` show progress of HMC sampling
* `treedepth` maximum depth of tree integrator

Typically `n_samples = n_adapts = 100` produces very accurate results.

"""
function get_site_stats(sites::Union{Int64, Array{Int64,1}, UnitRange{Int64}}, dat::ThrData; 
    n_samples = 1000, n_adapts=200, progress=false, treedepth=5) 

  D = 3;
  metric=DenseEuclideanMetric(D)

  logZmult = (4/3*π*dat.smax^3)
  retval = map(sites) do site
    # posteriors and transformation maps
    stmap, post, epost = 
    let Z = svd(dat.EE[:,:,site]), 
      Q = if size(Z.Vt, 1) == 2
        Q = zeros(Float64,(3,3))
        Q[:,1:2] = Z.Vt';
        Q[:,3] = cross(Q[:,1], Q[:,2]);
        Q
      elseif size(Z.Vt, 1) == 3
        Z.Vt'
      end

      omap = optimize(1e-3.*[1,1,1];autodiff=:forward) do s
        -lpost_simple(s, dat, site)
      end

      let alpha = norm(omap.minimizer)/2
        stmap = s->st_map(s, Q, alpha)
        post = s->stereo_post(s, alpha, dat, site, Q)
        epost = s->exp(stereo_post(s, alpha, dat, site, Q))
        stmap, post, epost
      end
    end

    #= samples, stats, initial_lpdf_val = get_samples_stmap() =#
    samples, stats = get_samples_nuts(post, [0.0, 0.0, 1.0], 3; 
                                      n_samples=n_samples, n_adapts=n_adapts,
                                      progress=progress)

    tree_integral = treeint.integrate(epost, vcat(samples'...); pad=0.21, maxdepth=treedepth)

    cart_samples = map(stmap, samples)
    mean_s = sum(cart_samples)/length(cart_samples)
    mean_Ethr = sum( (x->1 /sqrt(sum(x.^2))).(cart_samples) )/length(cart_samples)
    mean_d =    sum( (x->x./sqrt(sum(x.^2))).(cart_samples) )/length(cart_samples)
    mean_thr = sum( (x->1 ./ (dat.EE[:,:,site]*x)).(cart_samples) )/length(cart_samples)
            # sqrt(sum(x.^2))). (cart_samples) )/length(cart_samples)

    #= integrals = treeint.integrate(tree_integrand, vcat(samples'...); pad=0.21, maxdepth=4) =#
    #= @info "mean_s[1:3], integrals[2:4] ./integrals[1]: $(mean_s[1:3]), $(integrals[2:4] ./integrals[1])" =#

    return tree_integral/logZmult, mean_s, mean_Ethr, mean_d, cart_samples, mean_thr
  end
  retval

end 


function get_site_stats_path(site, dat::ThrData; 
        λs=[0.0001, 0.001, 0.01, 0.1, 1.0], 
        n_adapts=200, n_samples=1000)

  dfun = (s,l) -> lpost(s,Dual(l,1),dat,site).partials[1]
  epost = s -> exp(lpost(s, 1, dat, site))


  max_retry = 5

  Es = Array{Float64,1}(undef, length(λs))
  mean_s = Array{Float64,1}(undef, 3)
  mean_d = Array{Float64,1}(undef, 3)
  mean_Ethr = zero(Float64)

  for n_λ = 1:length(λs) 
      λ = λs[n_λ]
      retry = 0
      while true
          samples, stats, initial_lpdf_val = get_samples(dat, site, λ; 
                                                         n_samples=n_adapts+n_samples, 
                                                         progress=false, n_adapts=n_adapts)

          Es[n_λ] = mapreduce(s->dfun(s,λ),+,samples)/length(samples)
          
          if λ == 1.0
              mean_s = sum(samples)./length(samples)
              mean_Ethr = sum( (x-> (1 ./ (x.^2 |> sum |> sqrt))).(samples))/length(samples)
              mean_d =    sum( (x-> (x ./ (x.^2 |> sum |> sqrt))).(samples))/length(samples)
          end
          if !isinf(Es[n_λ]) || retry >= max_retry
              if retry >= max_retry
                  println("Warning: failed to calculate Es with lambda = $(λ)")
              end
              break
          end
          retry = retry + 1 
      end
  end
  logquot = sum( (λs[2:end]-λs[1:end-1]).*(Es[1:end-1] + Es[2:end]))/2;
  Z = exp(logquot)/(4/3*π)*(sqrt(2*π)*0.35)^3
  return Z, mean_s, mean_Ethr, mean_d
end

## Pessimistic model comparison with lognormal distribution comparison 
#
struct LogComparisonDensity
    data::Array{Number,1}
end

import LogDensityProblems.logdensity
import LogDensityProblems.dimension
import LogDensityProblems.capabilities


function logdensity(p::LogComparisonDensity, θ) 
    mapreduce(+,p.data) do x
        let μ = θ[1], σ =θ[2]
            if σ < 0.0
                -Inf
            else
                y = - log(x*σ*sqrt(2π)) - (log(x)-μ)^2/(2*σ^2)
            end
        end
    end
end

dimension(p::LogComparisonDensity) = 2
capabilities(::Type{LogComparisonDensity}) = LogDensityProblems.LogDensityOrder{0}()

function comparisonMLH(data)
    ℓπ=LogComparisonDensity(Array{Number,1}(data))
    metric = DenseEuclideanMetric(2)
    hamiltonian = Hamiltonian(metric, ℓπ, ForwardDiff)
    initial_θ = [0.5, 0.5]
    integrator = Leapfrog(find_good_stepsize(hamiltonian, initial_θ))
    proposal = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
    n_samples=1000
    n_adapts=1000

    samples, _ = sample(hamiltonian, 
        proposal,
        initial_θ, 
        n_samples+n_adapts, 
        StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator)),
        n_adapts;
        progress=false, verbose=false, 
        drop_warmup=true)

    treeint.integrate(
        θ->exp(logdensity(ℓπ, θ)), 
        Matrix(hcat(samples...)'); maxdepth=10, pad=0.51)
end


  """
      get_site_angle(site, meas_data)

  Returns the max angle of electric field vectors at the site
  """
  function get_site_angle(site, meas_data::ThrData) # {{{
    ONE_MINUS_EPS=1.000001 # Scale vectors so that they e.e/|e||e| is definitely \leq 1
    EE = meas_data.EE[:,:,site]
    n_E = sqrt.(sum(EE.^2; dims=2))*ONE_MINUS_EPS

    M=0.0

    for m = 1:size(EE,1)
      for n = (m+1):size(EE,1)
        M = max(acosd(sum(EE[m,:].*EE[n,:]) / (n_E[m]*n_E[n])), M)
      end
    end

    return M
  end # }}}

# }}}
