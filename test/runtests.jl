
#= using Revise =#

module TestTreeint
include("../src/treeint.jl")
import .treeint: integrate, gettree, Node
using Test

export testme

function testme()
    n_samples = 1000;

    fun2(x) = exp(-0.5*sum(x.*x))/(sqrt(2*pi)^2)
    fun2_2(x) = vcat(exp(-0.5*sum(x.*x))/(sqrt(2*pi)^2), exp(-0.5*sum(x.*x))/(sqrt(2*pi)^2)*x)
    fun3(x) = exp(-0.5*sum(x.*x))/(sqrt(2*pi)^3)
    samples = [randn(n_samples,i) for i in 1:3]

    # Precompile
    integral_2 = treeint.integrate(fun2, samples[2]; maxdepth = 5, pad = 0.51)
    integral_2_2 = treeint.integrate(fun2_2, samples[2]; maxdepth = 5, pad = 0.51)
    E_samples = [sum(S)/n_samples for S in samples]
    integral_3 = treeint.integrate(fun3, samples[3]; maxdepth = 5, pad = 0.51)

    print("2D integral = $(round(integral_2; digits=6)), error = $(1-integral_3)\n")
    print("2D integral with expectation (tree, samples): ");
    println(integral_2_2); print(" "); println(E_samples)
    print("3D integral = $(round(integral_3; digits=6)), error = $(1-integral_3)\n")

    @test abs(integral_2-1.0) < 5e-3
    @test abs(integral_3-1.0) < 5e-3

    tree = gettree(samples[2]; maxdepth=5, pad = 0.51)
    integral_2_sametree = integrate(fun2, tree)

    print("2D integral = $(round(integral_2_sametree; digits=6)), error = $(1-integral_3)\n")
end

end

module TestLTD


include("../src/LTDTMS.jl")
import BSON # TODO: get rid of BSON or not...
import .LTDTMS.Lpdfs: ThrData
import .LTDTMS: makeThrData 
import .LTDTMS: get_site_stats, get_site_stats_path

function testme() # TODO: Tests only that code runs but doesn't look at results

    inputdata = BSON.load("inputdata.bson")
    dat = makeThrData(inputdata)

    λ = 10.0 .^(-4:1:0)

    function tree_logint_to_Z(l)
        exp(l)*inputdata[:resolution]^3/(4/3*π/inputdata[:Ethrprior]^3)
    end

    function logquot_to_Z(l)
        exp(l)*inputdata[:resolution]^3/(4/3*π/inputdata[:Ethrprior]^3)*(sqrt(2*π)*0.35/inputdata[:Ethrprior]^3)
    end

    sites = [1,130,153,158,160,163]
    sites_small = [1,130]
    nuts=[true]


    @info "Precompile run"
    @time stats = get_site_stats(sites_small, dat; n_samples=1000, n_adapts=200)

    precompiled_Z=[tree_logint_to_Z(stats[n][1]) for n in 1:length(sites_small)]

    @info "Calculating Z"
    n_samples = [500, 500]

    @time Zcoeffs_rounds = map(1:length(n_samples)) do round
        sites = 1:size(inputdata[:EE],3)
        stats = get_site_stats(sites, dat; n_samples=n_samples[round], n_adapts=50)

        N = 1:length(sites)
        Dict(:Z =>[tree_logint_to_Z(stats[n][1]) for n in N],
             :d => [stats[n][4] for n in N],
             :E => [stats[n][3] for n in N],
             :s => [stats[n][2] for n in N])
    end


    Zcoeffs_mat = hcat(map(x->x[:Z],Zcoeffs_rounds)...)
    map(1:length(sites)) do site
        @info "Coefficients $(site): octree: $(Zcoeffs_mat[site,:])\n"
    end

    Zcoeffs = Zcoeffs_rounds[1][:Z]
    Z_max, Z_i = findmax(Zcoeffs)
    @info "maxind = $(Z_i)"

    for site = [1,Z_i]
        Z = Zcoeffs[site]
        stats_path = get_site_stats_path(sites[site], dat; λs=10.0 .^(-4:0.5:0), n_samples=2000, n_adapts=200)
        Z_path = logquot_to_Z(stats_path[1])
        @info "Coefficients $(site): octree: $(Z) | path: $(Z_path) | quotient = $(Z/Z_path)\n"
    end
    Zcoeffs_rounds

end

end

import .TestTreeint

import .TestLTD
@info "Testing LTD"
TestLTD.testme()

@info "Testing tree integrator"
TestTreeint.testme()


