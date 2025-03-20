using gcdyn, JLD2
import JSON

EVENT_MAPPING = Dict(
	"birth" => :birth,
	"death" => :unsampled_death,
	"mutation" => :type_change,
	"sampling" => :sampled_survival,
	"survival" => :unsampled_survival,
	"root" => :root
)

function main()
	trees = map(TreeNode, JSON.parsefile("out/trees/trees.json"))
	save_object("out/trees/trees.jld2", trees)
end

function gcdyn.TreeNode(json_tree::Dict)
	root = TreeNode(
		EVENT_MAPPING[json_tree["event"]],
		json_tree["time"],
		json_tree["affinity"]
	)

	for child in json_tree["children"]
		attach!(root, TreeNode(child))
	end

	return root
end

main()