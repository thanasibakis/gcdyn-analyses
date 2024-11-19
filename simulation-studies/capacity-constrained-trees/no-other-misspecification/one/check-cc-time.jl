using gcdyn, JLD2, StatsPlots

function main()
	trees = load_object("out/trees/trees-unpruned.jld2")
	population_curves = map(compute_population_curve, trees)

	times = map(compute_carrying_capacity_time, population_curves)
	histogram(times; label=nothing)
	title!("Times to reach carrying capacity")
	xlabel!("Time")
	mkpath("out/visualizations")
	savefig("out/visualizations/carrying-capacity-time.png")
end

function compute_carrying_capacity_time(population_curve; capacity=1000)
	idx = findfirst(x -> x[2] == capacity, population_curve)

	if isnothing(idx)
		return NaN
	end

	return population_curve[idx][1]
end

function compute_population_curve(tree)
	birth_times = sort([0; [node.time for node in PreOrderTraversal(tree) if node.event == :birth]])
	population_sizes = collect(1:length(birth_times))

	death_times = sort([node.time for node in PreOrderTraversal(tree) if node.event == :unsampled_death])

	map(death_times) do dt
		population_sizes[findall(>=(dt), birth_times)] .-= 1
	end

	return zip(birth_times, population_sizes) |> collect
end

function plot_population_curve(population_curve)
	plot(population_curve; xlabel="Time", ylabel="Population size", legend=nothing)

	capacity_time = compute_carrying_capacity_time(population_curve)
	vline!([capacity_time]; color=:red, label="Capacity time")
	title!("Carrying capacity at t=$(round(capacity_time, digits=1))")
end