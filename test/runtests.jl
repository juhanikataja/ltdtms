module TestLTD
using Test
using ProgressMeter


#= include("../src/LTDTMS.jl") =#
import BSON # TODO: get rid of BSON or not...
import LTDTMS.Lpdfs: ThrData
import LTDTMS: makeThrData 
import LTDTMS: get_site_stats, get_site_stats_path

function testme() # TODO: Tests only that code runs but doesn't look at results

    inputdata = BSON.load("inputdata.bson")
    dat = makeThrData(inputdata)

    thr_predict =  [
        0.24542675770366795,
        0.2869468039142507,
        0.2778941393656671,
        0.24053035100750988,
        0.25103365145337314,
        0.2415736484728888]

    sites = 1:size(inputdata[:EE],3)

    @info "Computing normalizing coefficients over $(length(sites)) sites"
    n_samples = 500
    tree_depth = 5
    p = Progress(length(sites))

    Zcoeffs = 
        let n_samples=500, 
            tree_depth=5,
            N = 1:length(sites),
            stats = map(sites) do site
                stat = get_site_stats(site, dat; n_samples=n_samples, n_adapts=100, treedepth=tree_depth)
                next!(p)
                stat
            end
            finish!(p)
            Dict(:Z =>[stats[n][1] for n in N],
                :d => [stats[n][4] for n in N],
                :E => [stats[n][3] for n in N],
                :s => [stats[n][2] for n in N],
                :thr => [stats[n][6] for n in N])
        end

    Z = Zcoeffs[:Z]
    Z_max, Z_i = findmax(Z)
    thr_predict_test = sum(Zcoeffs[:thr].*Z / sum(Z))

    @test Z_i == 158
    @test abs(Z_max - 4.759216449419632e6)/4.759216449419632e6 < 5e-2
    @test maximum(abs.(thr_predict-thr_predict_test)) < 5e-3

end

end

import .TestLTD
@info "Testing LTD"
TestLTD.testme()

