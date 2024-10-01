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
    type_space => [-2.4270176906430416, -1.4399117849363843, -0.6588015552361666, -0.13202968692343608, 0.08165101396850624, 0.7981793588605735, 1.3526378568771724, 2.1758707012574643],
	discretization_table => Dict([0.5, 1.0] => 0.7981793588605735, [-1.0, -0.5] => -0.6588015552361666, [2.0, Inf] => 2.1758707012574643, [0.0, 0.5] => 0.08165101396850624, [-2.0, -1.0] => -1.4399117849363843, [-Inf, -2.0] => -2.4270176906430416, [-0.5, 0.0] => -0.13202968692343608, [1.0, 2.0] => 1.3526378568771724),
	Γ => [-0.16502375081412976 0.12994973731821127 0.022865359830672522 0.008897832332297906 0.002690042333020297 0.0003103894999638804 0.0003103894999638804 0.0; 0.3099577571629828 -0.6374011568269109 0.14516382453614948 0.1034292249820065 0.05179709193676243 0.01963010809068385 0.0069282734437707705 0.000494876674555055; 0.10890059617389898 0.2948535009614057 -0.8203502457062107 0.17439505849546558 0.12842051435601295 0.06703550849383876 0.04237876973748427 0.00436629748810444; 0.04422242364632499 0.18529395609274632 0.1646835052530564 -0.802807075425592 0.17428837554728083 0.12846514018525176 0.09304718097529918 0.012806493725632575; 0.014990735730480613 0.10290515464986172 0.12429818376523509 0.16536655352686427 -0.8027851290666755 0.1703634654370245 0.1881649641169702 0.036696071840239; 0.0026776913807438807 0.039400316030945674 0.06598596616833134 0.11188924698108359 0.1683120296467582 -0.7910665393397636 0.2987538526229958 0.10404743650890508; 0.00037040077098561254 0.0077784161906978636 0.02049550932787056 0.05111530639601453 0.10408261664695713 0.14778990762325941 -0.6373362599425774 0.30570410298679224; 0.0 0.0 0.0014746530543487085 0.003041471924594211 0.009953908116853783 0.022580624894714598 0.12976946878268636 -0.16682012677319766],
	
    println("Reading trees...")

	trees = map(keys(TYPE_SPACES) |> collect) do name
		treeset::Vector{TreeNode{Float64}} = load_object("trees.jld2")
		discretization_table = TYPE_SPACES[name][:discretization_table]

		map(treeset) do tree
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

			tree
		end

		name => treeset
	end |> Dict

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
