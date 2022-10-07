module Lpdfs 

LAMBDA_POWER = 1

export lpost, ThrData, lpost_simple, lpost_simple_tes

"""
  Collects a motor threshold object that holds measured thresholds `MT`,
  electric field data `EE`, noise coeff `K`, inverse of prior threshold for
  electric field `smax = 1/Ethrprior`, and resolution `h`.

  # Constructors

  `ThrData(MT::Array{Float64,1}, E::Array{Float64,3}, K::Float64, Ethrprior::Float64, h::Float64)`
"""
struct ThrData
  MT::Array{Float64,1}
  EE::Array{Float64,3}
  K::Float64
  smax::Float64
  # h::Float64

  # ThrData(MT::Array{Float64,1}, E::Array{Float64,3}, K::Float64, Ethrprior::Float64, h::Float64) = 
  ThrData(MT::Array{Float64,1}, E::Array{Float64,3}, K::Float64, Ethrprior::Float64) = 
  begin
    return if size(MT,1) == size(E,1) && Ethrprior > 0.0 && K > 0.0
      new(MT, E, K, 1/Ethrprior)
    else
      error("Incompatible input data")
      new()
    end
  end
end

function ltanhstep_k(x;k=1)
  return if (-k*x > 20)
    -2*k*x
  else
    -log(1+exp(-2*k*x))
  end
end

function ltanhstep(x;k=1,N=1)
  (-1)^(N+1)*(ltanhstep_k(x;k=k)^N);
end

function lbox(x;a=0.25,b=0.75,k=[1,1],N=[1,1])
  ltanhstep(x-a;k=k[1],N=N[1]) + ltanhstep(b-x;k=k[2],N=N[2])
end

function sq(x)
  return sum(x.*x)
end


function hit(s, T, E, K)
  lπ  = 0
  Es = E * s
  #= lπ  = mapreduce(+, 1:size(T,1)) do n =#
  for n = 1:size(T,1)
    y = Es[n]
    x = T[n]*Es[n]
    lπ += log(abs(y)) - (x - 1)^2 / (2*K^2) - log(sqrt(2*pi)*K)
  end
  return lπ
end

function hit(s, MT, EE, lambda, K, smax, n) # {{{
  exponent = 0
  for m = 1:length(MT)
    exponent = exponent - lambda^(2*LAMBDA_POWER)*((MT[m]*sum(EE[m,:, n].*s)-1))^2 / (2*K^2)
    exponent = exponent + log(abs(sum(EE[m,:,n].*s)))*lambda^LAMBDA_POWER 
    exponent = exponent - log(sqrt(2*pi)*K)*lambda^LAMBDA_POWER  # TODO: this is possibly required so that likelihood pdf is a probability pdf
  end
  return exponent
end # }}}

function hit_simple(s, MT, EE, K, smax, n) # {{{
  lπ = 0
  Es = EE[:,:,n] * s
  #= lπ = mapreduce(+, 1:size(MT,1)) do m =#
  for m = 1:size(MT,1)
    y = Es[m]
    x = MT[m]*y
    lπ += log(abs(y)) - (x - 1)^2 / (2*K^2) - log(sqrt(2*pi)*K)
  end
  return lπ
end 

function ref_lpdf(s, lambda, K, smax) # {{{
  # This corresponds to $f(s) = exp\{-\frac 1 {2 (0.35*smax)^2} |s|^2\}$, $\int_{\reals^3} f(s) ds = (\sqrt{2\pi} 0.35 smax)^3$
  exponent = - (1-lambda)^(2*LAMBDA_POWER)*(sq(s)/(2*(0.35*smax)^2)) 
  return exponent
end # }}}

function z0(smax) # {{{
  return (sqrt(2*pi)*0.35*smax)^3
end# }}}

function prior_s(s, lambda, smax)
  return lambda^(LAMBDA_POWER*4)*ltanhstep(smax-sqrt(sq(s)); k=1000, N=4)
end

function prior_s_simple(s, smax)
  return ltanhstep(smax-sqrt(sq(s)); k=1000, N=4)
end

function lpost(s, lambda, dat::ThrData, n) # {{{
  exponent = 0
  exponent = exponent + hit(s, dat.MT, dat.EE, lambda, dat.K, dat.smax, n)
  
  return exponent + ref_lpdf(s, lambda, dat.K, dat.smax) + prior_s(s, lambda, dat.smax)
end # }}}

function lpost_simple(s, dat::ThrData, n)
  exponent = hit_simple(s, dat.MT, dat.EE, dat.K, dat.smax, n) + prior_s_simple(s, dat.smax)
  return exponent
end

function lpost_simple(s, T,E, K, smax)
    lπ = hit(s, T,E, K) + prior_s_simple(s, smax)

end


end 
