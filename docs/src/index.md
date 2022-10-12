# LTDTMS.jl Documentation

LTDTMS.jl is an octree-based integrator for location-threshold-direction TMS
(LTDTMS) localization model.[^1]

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
