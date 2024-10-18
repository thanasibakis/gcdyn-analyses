using gcdyn, JLD2

function main()
	STOP_CODONS = ["TAA", "TAG", "TGA"]
	invalid_trees = Vector{String}()

	for germinal_center_dir in readdir("data/jld2-with-sequences/"; join=true)
		for treefile in readdir(germinal_center_dir; join=true)
			tree::TreeNode{String} = load_object(treefile)

			for node in PostOrderTraversal(tree)
				sequence = node.type
				codons = [sequence[i:i+2] for i in 1:3:length(sequence)-2]
	
				if any(c in codons for c in STOP_CODONS)
					push!(invalid_trees, treefile)
					break
				end
			end
		end
	end

	open("out/trees-with-stop-codon.txt", "w") do io
		for treefile in invalid_trees
			println(io, treefile)
		end
	end
end

main()