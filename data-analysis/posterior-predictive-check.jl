println("Loading packages...")
using CSV, gcdyn, DataFrames, Distributions, JLD2, LinearAlgebra, Logging, Optim, Random, StatsBase

# rand_tree gives info I won't need here
Logging.disable_logging(Logging.Info)

function main()
	seed = "SEED" in keys(ENV) ? parse(Int, ENV["SEED"]) : 1
	Random.seed!(seed)

	out_path = "out/seed-$seed/"
	mkpath(out_path)

	# Generated in the `type-spaces` directory at the repository root
	type_space = [-2.4270176906430416, -1.4399117849363843, -0.6588015552361666, -0.13202968692343608, 0.08165101396850624, 0.7981793588605735, 1.3526378568771724, 2.1758707012574643]
	Γ = [-0.0002530983171703816 0.00020380576101734844 1.801074167130056e-5 1.3271072810431992e-5 8.531403949563423e-6 4.739668860868569e-6 4.739668860868569e-6 0.0; 0.19660874781462317 -0.2170049358114207 0.016807159207661463 0.003063805063896621 0.00026261186261971036 0.0001750745750798069 8.753728753990345e-5 0.0; 0.10942069146591266 0.1497958744948011 -0.2897281049271702 0.02498789829149448 0.005392125420796177 0.0 0.0 0.0001315152541657604; 0.08965409890867519 0.05234201788572127 0.10047985762801112 -0.2823105623314204 0.03321300733310828 0.006621580575904499 0.0 0.0; 0.07521838264330194 0.024332389553068143 0.0641123127228144 0.1303448751577219 -0.31712877836592124 0.019183211680842104 0.003937606608172853 0.0; 0.07455086435585298 0.018816924897510967 0.013978287066722434 0.04533982856257405 0.1075252851286341 -0.28763013771909623 0.0274189477078017 0.0; 0.06323847403404087 0.021986886667464543 0.008166557905058259 0.009213552508270856 0.0452301668587842 0.08836634451114321 -0.24394974254853516 0.00774776006377322; 0.08452697029483416 0.01470034265997116 0.00367508566499279 0.01102525699497837 0.01102525699497837 0.01102525699497837 0.18742936891463227 -0.32340753851936554]
	discretization_table = Dict([0.5, 1.0] => 0.7981793588605735, [-1.0, -0.5] => -0.6588015552361666, [2.0, Inf] => 2.1758707012574643, [0.0, 0.5] => 0.08165101396850624, [-2.0, -1.0] => -1.4399117849363843, [-Inf, -2.0] => -2.4270176906430416, [-0.5, 0.0] => -0.13202968692343608, [1.0, 2.0] => 1.3526378568771724)

	# Compute counts of observed leaves
	observed_leaves = map(readdir("data/jld2-with-affinities"; join=true)) do germinal_center_dir
		tree_path = readdir(germinal_center_dir; join=true)[1]
		tree::TreeNode{Float64} = load_object(tree_path)

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

		map(leaf -> leaf.type, LeafTraversal(tree))
	end

	germinal_center_sizes = map(length, observed_leaves)

	# Draw posterior predictive trees
	println("Reading samples...")
	posterior_samples = CSV.read(joinpath(out_path, "posterior-samples.csv"), DataFrame)

	println("Sampling trees...")
	result = map(posterior_samples[1:5:end, :] |> eachrow) do row
		trees = map(germinal_center_sizes) do gc_size
			ρ = gc_size / 1000
			model = SigmoidalBranchingProcess(row[10], row[11], row[9], row[12], row[13], row[14], Γ, ρ, 0, type_space)

			rand_tree(model, 15, type_space[5])
		end

		tree_info = map(trees |> enumerate) do (tree_i, tree)
			node_info = map(PostOrderTraversal(tree)) do node
				(
					iteration = row[1],
					tree = tree_i,
					node = node.event == :sampled_survival ? "leaf" : "internal",
					affinity = node.type
				)
			end

			DataFrame(node_info)
		end

		vcat(tree_info...)
	end |> (dfs -> vcat(dfs...))

	CSV.write(joinpath(out_path, "posterior-predictive-affinities.csv"), result)

	println("Done!")
end

main()
