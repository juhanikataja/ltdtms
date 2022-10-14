# LTDTMS.jl: A probabilistic LTDTMS-model integrator for TMS activation localization

LTDTMS.jl is an octree-based integrator for location-threshold-direction TMS
(LTDTMS) localization model.[^1]

The methods [`get_site_stats`](@ref) calculate the following kinds of integrals in $\mathbb R^3$:
$\int_B \pi(t_1,\dots,t_N|s, E_1, \dots, E_N) ds,$
where 

$\pi(t_1,\dots,t_N|s, E_1,\dots,E_N) = \prod_{k=1}^N \frac{|E_k\cdot s|}{\sqrt{2\pi}K} e^{-\frac 1 {2K^2}(t_k E_k\cdot s -1)^2},$ 

and $B=\{s\in\mathbb R^3: |s| < \frac 1 {E_{\text{thr,prior}}}\}$, $t_k$ is
threshold measurement and $E_k$ is an electric field vector, $K$ is noise
hyperparameter, and $s$ is the preferred direction divided by threshold field.

In practice the above integral is calculated over multiple sets of electric
field values each corresponding to candidate activation location, and thus they
are collected to a $N_{thr}\times 3 \times N_{site}$ array.

[^1]: 
    Kataja, Juhani; Soldati, Marco; Matilainen, Noora; Laakso, Ilkka, *A probabilistic transcranial magnetic stimulation localization method*, Journal of Neural Engineering, Aug. 2021, Vol. 18, No. 4, IOP Publishing, p. 0460f3

For examples see, `examples/`. Particularly, `examples/matlab` contains
skeleton code for building matlab interoperable workflow (using `jlcall.m` and
`MATDaemon.jl`).

## Main methods and structures

```@docs
get_site_stats(T,E,K,Ethrprior; kwargs...)
get_site_stats(sites::Union{Int64, Array{Int64,1}, UnitRange{Int64}}, dat::ThrData; 
  n_samples = 1000, n_adapts=200, progress=false, treedepth=5) 
```

```@docs
ThrData
```


## Helper functions 
```@docs
get_site_stats_ml(T,E,K,Ethrprior; kwargs...)
```
