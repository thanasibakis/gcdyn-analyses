using gcdyn, CategoricalArrays, CSV, DataFrames, Distributions, JLD2, LinearAlgebra

function main()
	# Note that the 10x heavy and light chain sequences each have an extra character and need correcting
	tenx_data = CSV.read("support/10x-data.csv", DataFrame)
	tenx_data.sequence = chop.(tenx_data.nt_seq_H) .* chop.(tenx_data.nt_seq_L)
	tenx_data.time = tenx_data.var"time (days)"

	# Remove sequences observed after 20 days and also young sequences,
	# which are mostly the naive because germinal centers are still spawning
	filter!(:time => <=(20), tenx_data)
	filter!(:time => >(5), tenx_data)

	# Create an 8-element type space
	bin_cutoffs = [-Inf, -2, -1, -0.5, 0, 0.5, 1, 2, Inf]
	discretization_table = compute_manual_discretization_table(tenx_data.delta_bind_CGG, bin_cutoffs)
	type_space = values(discretization_table) |> collect |> sort
	tc_rate_matrix = compute_rate_matrix(tenx_data.sequence, discretization_table, type_space)

	mkpath("out/")
	open("out/type-space-8-elements.txt", "w") do io
		println(io, "type_space = ", repr(type_space))
		println(io, "discretization_table = ", repr(discretization_table))
		println(io, "Γ = ", repr(tc_rate_matrix))
	end

	# Create a 10-element type space
	bin_cutoffs = [-Inf, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, Inf]
	discretization_table = compute_manual_discretization_table(tenx_data.delta_bind_CGG, bin_cutoffs)
	type_space = values(discretization_table) |> collect |> sort
	tc_rate_matrix = compute_rate_matrix(tenx_data.sequence, discretization_table, type_space)

	open("out/type-space-10-elements.txt", "w") do io
		println(io, "type_space = ", repr(type_space))
		println(io, "discretization_table = ", repr(discretization_table))
		println(io, "Γ = ", repr(tc_rate_matrix))
	end
end

function compute_quantile_discretization_table(affinities; type_space_size=5)		
	# Create bins from evenly spaced quantiles, then discretize types to the medians of their bins
	cutoffs = quantile(affinities, 0:(1/type_space_size):1)
	bin_table = DataFrame(
		type=affinities,
		bin=cut(affinities, cutoffs; extend=true)
	)
	bin_table = DataFrames.transform(groupby(bin_table, :bin), :type => median => :binned_type)
	bin_table = select(bin_table, Not(:type)) |> unique
	
	parse_interval(row) = parse.(Float64, split(convert(String, row.bin)[2:end-1], ", "))
	
	Dict(parse_interval(row) => row.binned_type for row in eachrow(bin_table))
end

function compute_manual_discretization_table(affinities, cutoffs)		
	# Create bins from given cutoffs, then discretize types to the medians of their bins
	bin_table = DataFrame(
		type=affinities,
		bin=cut(affinities, cutoffs; extend=true)
	)
	bin_table = DataFrames.transform(groupby(bin_table, :bin), :type => median => :binned_type)
	bin_table = select(bin_table, Not(:type)) |> unique
	
	parse_interval(row) = parse.(Float64, split(convert(String, row.bin)[2:end-1], ", "))
	
	Dict(parse_interval(row) => row.binned_type for row in eachrow(bin_table))
end

function get_discretization(affinity, discretization_table)
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

function compute_rate_matrix(starting_sequences, discretization_table, type_space)
	type_change_history = let
		command = pipeline(`support/simulate-s5f-mutations.py`; stdin=IOBuffer(join(starting_sequences, "\n")))
		output = read(command, String) |> strip
		rows = split(output, "\n")
		mutation_histories = split.(rstrip.(rows, ';'), ";")

		map(mutation_histories) do mutations
			mutations = [parse.(Float64, split(mutation, ':')) for mutation in mutations]
			[(get_discretization(from_affinity, discretization_table), duration, get_discretization(to_affinity, discretization_table)) for (from_affinity, duration, to_affinity) in mutations]
		end
	end

	rate_matrix = zeros(length(type_space), length(type_space))
	duration_times = zeros(length(type_space))

	for history in type_change_history
		for (from_type, duration, to_type) in history
			i = findfirst(==(from_type), type_space)
			j = findfirst(==(to_type), type_space)
			
			if i != j
				rate_matrix[i, j] += 1
			end

			duration_times[i] += duration
		end
	end

	rate_matrix ./= duration_times
	rate_matrix[diagind(rate_matrix)] = -sum.(eachrow(rate_matrix))

	rate_matrix
end

main()
