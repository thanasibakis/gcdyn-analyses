println("Loading packages...")
using CSV, gcdyn, DataFrames, Distributions, JLD2, LinearAlgebra, Optim, Random, StatsBase, Turing

@model function SigmoidalModel(trees, Γ, type_space)
	# Keep priors on the same scale for NUTS
    θ ~ MvNormal(zeros(6), I)

    # Obtain our actual parameters from the proxies
    φ := [
        exp(θ[1] * 0.75 + 0.5),  # yscale
        exp(θ[2] * 0.75 + 0.5),  # xscale
        θ[3] * sqrt(2),          # xshift
        exp(θ[4] * 1.2 - 0.5),   # yshift
    ]
    μ := exp(θ[5] * 0.5)
    δ := exp(θ[6] * 0.5)

	if DynamicPPL.leafcontext(__context__) !== Turing.PriorContext()
		for tree in trees
			ρ = length(LeafTraversal(tree)) / 1000
			present_time = maximum(node.time for node in LeafTraversal(tree))

			sampled_model = SigmoidalBranchingProcess(φ[2], φ[3], φ[1], φ[4], μ, δ, Γ, ρ, 0, type_space)
			
			Turing.@addlogprob! loglikelihood(sampled_model, tree, present_time)
		end
	end
end

function load_tree(path, discretization_table)
	tree = load_object(path)::TreeNode{Float64}

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

function main()
	Random.seed!(1)

	out_path = "out/inference/one-sample-no-stop-codons/"
	mkpath(out_path)

	# Generated in the `type-spaces` directory at the repository root
	type_space = [-2.4270176906430416, -1.4399117849363843, -0.6588015552361666, -0.13202968692343608, 0.08165101396850624, 0.7981793588605735, 1.3526378568771724, 2.1758707012574643]
	discretization_table = Dict([0.5, 1.0] => 0.7981793588605735, [-1.0, -0.5] => -0.6588015552361666, [2.0, Inf] => 2.1758707012574643, [0.0, 0.5] => 0.08165101396850624, [-2.0, -1.0] => -1.4399117849363843, [-Inf, -2.0] => -2.4270176906430416, [-0.5, 0.0] => -0.13202968692343608, [1.0, 2.0] => 1.3526378568771724)
	Γ = [-0.0002530983171703816 0.00020380576101734844 1.801074167130056e-5 1.3271072810431992e-5 8.531403949563423e-6 4.739668860868569e-6 4.739668860868569e-6 0.0; 0.19660874781462317 -0.2170049358114207 0.016807159207661463 0.003063805063896621 0.00026261186261971036 0.0001750745750798069 8.753728753990345e-5 0.0; 0.10942069146591266 0.1497958744948011 -0.2897281049271702 0.02498789829149448 0.005392125420796177 0.0 0.0 0.0001315152541657604; 0.08965409890867519 0.05234201788572127 0.10047985762801112 -0.2823105623314204 0.03321300733310828 0.006621580575904499 0.0 0.0; 0.07521838264330194 0.024332389553068143 0.0641123127228144 0.1303448751577219 -0.31712877836592124 0.019183211680842104 0.003937606608172853 0.0; 0.07455086435585298 0.018816924897510967 0.013978287066722434 0.04533982856257405 0.1075252851286341 -0.28763013771909623 0.0274189477078017 0.0; 0.06323847403404087 0.021986886667464543 0.008166557905058259 0.009213552508270856 0.0452301668587842 0.08836634451114321 -0.24394974254853516 0.00774776006377322; 0.08452697029483416 0.01470034265997116 0.00367508566499279 0.01102525699497837 0.01102525699497837 0.01102525699497837 0.18742936891463227 -0.32340753851936554]
	
	# Read in the trees used to compute the MAP
	println("Reading trees...")
	germinal_center_dirs = readdir("data/jld2-with-affinities/"; join=true)

	filter!(germinal_center_dirs) do germinal_center_dir
		# Filter out germinal centers whose sampled sequences have a stop codon
		!(basename(germinal_center_dir) in ["btt-PR-2-01-1-LI-1D-GC", "btt-PR-2-01-1-RP-1A-GC", "btt-PR-2-01-10-LP-10B-GC", "btt-PR-2-01-2-RP-2A-GC", "btt-PR-2-01-3-LP-3B-GC", "btt-PR-2-01-3-RP-3A-GC", "btt-PR-2-01-4-LP-4B-GC", "btt-PR-2-01-5-LI-5D-GC", "btt-PR-2-01-6-RP-6A-GC", "btt-PR-2-02-10-RI-10C-GC", "btt-PR-2-02-10-RP-10A-GC", "btt-PR-2-02-12-RP-12A-GC", "btt-PR-2-02-7-RI-7C-GC", "btt-PR-2-02-7-RP-7A-GC", "btt-PR-2-03-1-LI-4A-GC", "btt-PR-2-03-1-LP-2A-GC", "btt-PR-2-03-2-RI-7B-GC", "btt-PR-2-03-6-LP-21B-GC", "btt-PR-2-03-7-LP-25A-GC", "btt-PR-2-03-8-RP-28A-GC", "btt-PR-2-04-1-LP-2B-GC", "btt-PR-2-04-1-RP-1A-GC", "btt-PR-2-04-2-RP-5A-GC", "btt-PR-2-04-5-RP-17A-GC", "btt-PR-2-04-6-LP-21A-GC", "btt-PR-2-04-6-RI-22A-GC", "btt-PR-2-04-6-RI-22B-GC", "btt-PR-2-04-7-LP-25B-GC"])
	end

	treeset = map(germinal_center_dirs) do germinal_center_dir
		load_tree(sample(readdir(germinal_center_dir; join=true)), discretization_table)
	end

	println("Computing initial MCMC state...")
	max_a_posteriori = optimize(SigmoidalModel(treeset, Γ, type_space), MAP(), NelderMead())

	open(joinpath(out_path, "map.txt"), "w") do f
		println(f, max_a_posteriori)
	end

	println("Sampling from posterior...")
	
	posterior_samples = sample(
		SigmoidalModel(treeset, Γ, type_space),
		NUTS(adtype=AutoForwardDiff(chunksize=6)),
		1000;
		init_params=max_a_posteriori
	) |> DataFrame

	CSV.write(joinpath(out_path, "samples-posterior.csv"), posterior_samples)

	println("Done!")
end

main()
