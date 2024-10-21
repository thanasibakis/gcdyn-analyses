println("Loading packages...")
using CSV, DataFrames, Dates, gcdyn, JLD2, LinearAlgebra, Optim, Random, Turing

@model function Model(trees, Γ, type_space, ρ, σ, present_time)
    # Keep priors on the same scale for NUTS
    θ ~ MvNormal(zeros(6), I)

    # Obtain our actual parameters from the proxies
    φ := [
        exp(θ[3] * 0.75 + 0.5),  # yscale
        exp(θ[1] * 0.75 + 0.5),  # xscale
        θ[2] * sqrt(2),          # xshift
        exp(θ[4] * 1.2 - 0.5),   # yshift
    ]
    μ := exp(θ[5] * 0.5)
    δ := exp(θ[6] * 0.5)

    if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
        sampled_model = SigmoidalBranchingProcess(φ[2], φ[3], φ[1], φ[4], μ, δ, Γ, ρ, σ, type_space)

        Turing.@addlogprob! loglikelihood(sampled_model, trees, present_time)
    end
end

function main()
    Random.seed!(1)

    # Generated in the `type-spaces` directory at the repository root
    type_space = [-2.4270176906430416, -1.6804633508256577, -1.2563382551385835, -0.6588015552361666, -0.13202968692343608, 0.08165101396850624, 0.7981793588605735, 1.2620888983735103, 1.685078638755301, 2.1758707012574643]
    discretization_table = Dict([0.5, 1.0] => 0.7981793588605735, [1.0, 1.5] => 1.2620888983735103, [-1.5, -1.0] => -1.2563382551385835, [-1.0, -0.5] => -0.6588015552361666, [2.0, Inf] => 2.1758707012574643, [0.0, 0.5] => 0.08165101396850624, [-Inf, -2.0] => -2.4270176906430416, [-0.5, 0.0] => -0.13202968692343608, [1.5, 2.0] => 1.685078638755301, [-2.0, -1.5] => -1.6804633508256577)
    Γ = [-0.00022915341966718325 0.00014487798846726877 3.5982768377491586e-5 2.3672873932560254e-5 1.2309894444931332e-5 8.522234615721691e-6 2.8407448719072307e-6 0.0 9.469149573024102e-7 0.0; 0.2683155738122384 -0.2997604213214599 0.025031753609314457 0.005171849919279847 0.0 0.0004137479935423877 0.00020687399677119385 0.0006206219903135816 0.0 0.0; 0.15594350314035088 0.09670239580211704 -0.2843573152235224 0.022825250180319515 0.007666496243771441 0.0005227156529844164 0.0 0.0006969542039792218 0.0 0.0; 0.10759660446297442 0.044343650106646634 0.09774245999483072 -0.2808431173420954 0.024102704712621745 0.006791369836153089 0.0001331641144343743 0.0 0.0 0.0001331641144343743; 0.09345290965901232 0.013507294815579924 0.048538409012083956 0.09916331071925749 -0.2989175324228338 0.037556868511612475 0.006698739705287605 0.0 0.0 0.0; 0.07963201509538133 0.010027735234233205 0.011207468791201818 0.060363033664894 0.1308521136937686 -0.3123344592074401 0.017007825446297493 0.003244267281663684 0.0 0.0; 0.07427917761992531 0.008657246808849103 0.009176681617380049 0.0135053050218046 0.042593654299537584 0.1031943819614813 -0.2742615789043396 0.01939223285182199 0.0034628987235396412 0.0; 0.06572358338075009 0.008178934820715567 0.009639458895843346 0.011976297416047794 0.011099982970971126 0.05958938226521341 0.10691036229935348 -0.28889166206027483 0.012268402231073349 0.003505257780306671; 0.05754228727577066 0.010376478033335693 0.01414974277273049 0.0037732647393947977 0.011319794218184392 0.010376478033335693 0.054712338721224565 0.10282146414850823 -0.2867681201940046 0.021696272251520085; 0.0501096659724595 0.0031318541232787187 0.009395562369836156 0.015659270616393595 0.0031318541232787187 0.018791124739672312 0.0031318541232787187 0.06263708246557438 0.10335118606819772 -0.26933945460196984]
    
    println("Reading trees...")

    trees::Vector{TreeNode{Float64}} = load_object("../trees.jld2")

	for tree in trees
        map_types!(tree) do affinity
            for (bin, value) in discretization_table
                if bin[1] <= affinity < bin[2]
                    return value
                end
            end
        
            if all(bin[2] <= affinity for bin in keys(discretization_table))
                return maximum(values(discretization_table))
            elseif all(affinity < bin[1] for bin in keys(discretization_table))
                return minimum(values(discretization_table))
            else
                error("Affinity $affinity not in any bin!")
            end
        end
    end

    num_treesets = 5
	num_trees_per_set = 52
    present_time = 15

    ρ = 0.1
    σ = 0

    mkpath("out/")
    save_object("out/trees.jld2", trees)

	println("Sampling from posteriors...")
    dfs = Vector{DataFrame}(undef, num_treesets)

    Threads.@threads for i in 1:num_treesets
        println("[$(Dates.format(now(), "mm/dd HH:MM"))] Sampling from posterior $i...")

        treeset = trees[(i - 1) * num_trees_per_set + 1:i * num_trees_per_set]
        model = Model(treeset, Γ, type_space, ρ, σ, present_time)

        max_a_posteriori = optimize(model, MAP(), NelderMead())

        dfs[i] = sample(
            model,
            NUTS(adtype=AutoForwardDiff(chunksize=6)),
            1000;
            init_params=max_a_posteriori
        ) |> DataFrame

        dfs[i].treeset .= i

        println("Finished sampling from posterior $i")
    end

    println("Exporting samples...")
    posterior_samples = vcat(dfs...)
    CSV.write("out/posterior-samples.csv", posterior_samples)

	println("Done!")
end

main()
