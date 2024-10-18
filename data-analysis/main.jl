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

function resample_tree(germinal_center_dir, discretization_table, current_parameters, beast_logdensities, Γ, type_space)
	treefiles = readdir(germinal_center_dir; join=true)[2:end] # Skip the initial tree
	weights = Vector{Float64}(undef, length(treefiles))

	Threads.@threads for i in eachindex(treefiles)
		tree = load_tree(treefiles[i], discretization_table)
		state = parse(Int, match(r"tree-STATE_(\d+)", basename(treefiles[i])).captures[1])
		present_time = maximum(node.time for node in LeafTraversal(tree))
		ρ = length(LeafTraversal(tree)) / 1000

		current_model = SigmoidalBranchingProcess(
			current_parameters[2], current_parameters[3], current_parameters[1], current_parameters[4], current_parameters[5], current_parameters[6], Γ, ρ, 0, type_space
		)

		weights[i] = exp(
			loglikelihood(current_model, tree, present_time) - beast_logdensities[germinal_center_dir][state]
		)
	end

	selected_name = sample(treefiles, Weights(weights))

	return selected_name, load_tree(selected_name, discretization_table)
end

function main()
	Random.seed!(1)

	out_path = "out/inference/"
	mkpath(out_path)

	# Generated in the `type-spaces` directory at the repository root
	type_space = [-2.4270176906430416, -1.4399117849363843, -0.6588015552361666, -0.13202968692343608, 0.08165101396850624, 0.7981793588605735, 1.3526378568771724, 2.1758707012574643]
	discretization_table = Dict([0.5, 1.0] => 0.7981793588605735, [-1.0, -0.5] => -0.6588015552361666, [2.0, Inf] => 2.1758707012574643, [0.0, 0.5] => 0.08165101396850624, [-2.0, -1.0] => -1.4399117849363843, [-Inf, -2.0] => -2.4270176906430416, [-0.5, 0.0] => -0.13202968692343608, [1.0, 2.0] => 1.3526378568771724)
	Γ = [-0.0002530983171703816 0.00020380576101734844 1.801074167130056e-5 1.3271072810431992e-5 8.531403949563423e-6 4.739668860868569e-6 4.739668860868569e-6 0.0; 0.19660874781462317 -0.2170049358114207 0.016807159207661463 0.003063805063896621 0.00026261186261971036 0.0001750745750798069 8.753728753990345e-5 0.0; 0.10942069146591266 0.1497958744948011 -0.2897281049271702 0.02498789829149448 0.005392125420796177 0.0 0.0 0.0001315152541657604; 0.08965409890867519 0.05234201788572127 0.10047985762801112 -0.2823105623314204 0.03321300733310828 0.006621580575904499 0.0 0.0; 0.07521838264330194 0.024332389553068143 0.0641123127228144 0.1303448751577219 -0.31712877836592124 0.019183211680842104 0.003937606608172853 0.0; 0.07455086435585298 0.018816924897510967 0.013978287066722434 0.04533982856257405 0.1075252851286341 -0.28763013771909623 0.0274189477078017 0.0; 0.06323847403404087 0.021986886667464543 0.008166557905058259 0.009213552508270856 0.0452301668587842 0.08836634451114321 -0.24394974254853516 0.00774776006377322; 0.08452697029483416 0.01470034265997116 0.00367508566499279 0.01102525699497837 0.01102525699497837 0.01102525699497837 0.18742936891463227 -0.32340753851936554]
	
	# Read in the trees used to compute the MAP
	println("Reading trees...")
	germinal_center_dirs = readdir("data/jld2-with-affinities/"; join=true)

	starter_treeset = map(germinal_center_dirs) do germinal_center_dir
		load_tree(joinpath(germinal_center_dir, "tree-STATE_10000000.jld2"), discretization_table)
	end

	println("Computing initial MCMC state...")
	max_a_posteriori = optimize(SigmoidalModel(starter_treeset, Γ, type_space), MAP(), NelderMead())

	open(joinpath(out_path, "map.txt"), "w") do f
		println(f, max_a_posteriori)
	end

	println("Sampling from posterior...")

	# Load the tree densities from BEAST
	beast_logdensities = map(germinal_center_dirs) do germinal_center_dir
		beast_logfile = filter(
			endswith(".log"),	
			readdir(joinpath("data/raw", basename(germinal_center_dir)); join=true)
		) |> first

		df = CSV.read(beast_logfile, DataFrame; header=5, select = [:state, :skyline])

		germinal_center_dir => Dict{Int, Float64}(zip(df.state, df.skyline))
	end |> Dict

	# Main SIR loop
	num_mcmc_iterations = 2 #1000
	chain = Matrix{Float64}(undef, num_mcmc_iterations+1, 6)
	chain[1, :] = max_a_posteriori.values[7:12]

	selected_trees = Dict{String, Int64}()

	for mcmc_iteration in 1:num_mcmc_iterations
		if mcmc_iteration % 10 == 0
			println("INFO: Starting SIR iteration $mcmc_iteration")
		end

		treeset = map(germinal_center_dirs) do germinal_center_dir
			selected_name, tree = resample_tree(germinal_center_dir, discretization_table, chain[mcmc_iteration, :], beast_logdensities, Γ, type_space)

			if haskey(selected_trees, selected_name)
				selected_trees[selected_name] += 1
			else
				selected_trees[selected_name] = 1
			end

			tree
		end

		parameter_samples::Chains = sample(
			SigmoidalModel(treeset, Γ, type_space),
			NUTS(adtype=AutoForwardDiff(chunksize=6)),
			10,
			init_params=chain[mcmc_iteration, :]
		)

		chain[mcmc_iteration+1, :] = parameter_samples.value[end, 7:12, 1]
	end

	posterior_samples = DataFrame(chain, names(max_a_posteriori.values[7:12])[1])
	posterior_samples.iteration = 0:num_mcmc_iterations
	CSV.write(joinpath(out_path, "samples-posterior.csv"), posterior_samples)

	open(joinpath(out_path, "selected-trees.txt"), "w") do f
		println(f, sort(selected_trees, by=x->x[2], rev=true))
	end

	println("Done!")
end

main()
