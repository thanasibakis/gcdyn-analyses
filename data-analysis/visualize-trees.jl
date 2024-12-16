ENV["GKSwstype"] = "100" # Headless plotting mode

using gcdyn, JLD2, Plots, Random, StatsBase

function main()
	# Generated in the `type-spaces` directory at the repository root
	discretization_table = Dict([0.5, 1.0] => 0.7981793588605735, [-1.0, -0.5] => -0.6588015552361666, [2.0, Inf] => 2.1758707012574643, [0.0, 0.5] => 0.08165101396850624, [-2.0, -1.0] => -1.4399117849363843, [-Inf, -2.0] => -2.4270176906430416, [-0.5, 0.0] => -0.13202968692343608, [1.0, 2.0] => 1.3526378568771724)
	
	out_path = "out/tree-visualizations/"
	mkpath(out_path)

	for germinal_center_dir in readdir("data/jld2-with-affinities/"; join=true)
		gc_name = basename(germinal_center_dir)
		directory_name = joinpath(out_path, gc_name)
		mkpath(directory_name)

		for i in (5:5:45) * 1000000
			tree::TreeNode{Float64} = load_object(joinpath(germinal_center_dir, "tree-STATE_$i.jld2"))

			# Correct 15-day trees to have leaves at 15 days instead of the incorrect 20
			for node in PostOrderTraversal(tree)
				node.time = 15/20 * node.time
			end

			# Don't prune self loops when binning here. We want to visualize when the nucleotide-level mutations occurred too
			map_types!(tree; prune_self_loops=false) do affinity
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

			p = plot(
				tree;
				colorscheme=:diverging_bkr_55_10_c35_n256,
				midpoint=0.08165101396850624,
				reverse_colorscheme=true,
				title="$gc_name STATE_$i",
				dpi=500,
				size=(1000, 700),
				legendtitle="Affinity bin"
			)

			png(p, joinpath(directory_name, "tree-STATE_$i.png"))
		end
	end
end

main()
