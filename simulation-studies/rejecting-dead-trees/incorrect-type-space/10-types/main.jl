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
    Γ = [-0.004184455772426456 0.0023911175842436893 0.0006897454569933719 0.00036786424372979836 0.00022991515233112398 0.00036786424372979836 4.5983030466224795e-5 9.196606093244959e-5 0.0 0.0; 0.23057642225946032 -0.2680450908766226 0.02882205278243254 0.007823128612374547 0.0 0.0 0.0008234872223552154 0.0 0.0 0.0; 0.15730291710861286 0.10701135947533037 -0.2930523095172475 0.02268792073681916 0.00605011219648511 0.0 0.0 0.0 0.0 0.0; 0.11134713738477038 0.050340851384685456 0.10132162884629489 -0.2945793040773331 0.025170425692342728 0.006185952076931687 0.00021330869230798923 0.0 0.0 0.0; 0.09621337160500798 0.012581748594501044 0.038337328070303184 0.10257825618810851 -0.30092582226612496 0.04277794522130355 0.0084371725869007 0.0 0.0 0.0; 0.07969236719501899 0.010410822464046026 0.011898082816052601 0.06804216110430082 0.13806733601127708 -0.3310393666841302 0.01908650785075105 0.0038420892426836526 0.0 0.0; 0.06877925548505934 0.012776332288246317 0.010646943573538598 0.01639629310324944 0.04897594043827755 0.09880363636243818 -0.28193106582730204 0.02171976489001874 0.003619960815003123 0.00021293887147077193; 0.06417591592224327 0.008356239052375425 0.008690488614470443 0.007353490366090374 0.014706980732180748 0.060833420301293094 0.11097085461554565 -0.29447386420571 0.017380977228940886 0.002005497372570102; 0.04741890341898872 0.011591287502419464 0.010537534093108604 0.008430027274486883 0.007376273865176023 0.016860054548973765 0.04847265682829958 0.10959035456832948 -0.27502963983013456 0.014752547730352046; 0.02171670823336106 0.013030024940016637 0.004343341646672212 0.034746733173377696 0.017373366586688848 0.0 0.013030024940016637 0.07818014964009981 0.09555351622678866 -0.27797386538702157]
    
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