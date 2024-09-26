println("Loading packages...")
using CSV, DataFrames, gcdyn, JLD2, LinearAlgebra, Optim, Random, Turing

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
    
	println("Generating trees...")

    # Generated in the `type-spaces` directory at the repository root
    type_space = [-2.4270176906430416, -1.4399117849363843, -0.6588015552361666, -0.13202968692343608, 0.08165101396850624, 0.7981793588605735, 1.3526378568771724, 2.1758707012574643]
    Γ = [-0.004619127041534006 0.0033511313830736906 0.0004528555923072555 0.00027171335538435327 0.0003622844738458044 9.05711184614511e-5 9.05711184614511e-5 0.0; 0.2020753168875446 -0.2209145269997298 0.013881523240557529 0.0031729195978417207 0.0009915373743255376 0.00019830747486510755 0.0005949224245953227 0.0; 0.1102975726069481 0.15072536814415619 -0.28848747576279454 0.02263077684963278 0.004614041881963965 0.00021971628009352211 0.0 0.0; 0.0921587483881802 0.051134186307591235 0.0984589489934135 -0.2857067716326731 0.03618952440680526 0.00776536353668291 0.0 0.0; 0.0865376168167631 0.02461627223233437 0.06534732061676392 0.13285173725388705 -0.3320658991341189 0.01827187217245438 0.004441080041915994 0.0; 0.07206014315123183 0.021894379777512914 0.015304809164863396 0.048677795816023856 0.10096922712930713 -0.2835641031378857 0.024657748098946584 0.0; 0.05828029654047501 0.020704842192010856 0.00792407540681897 0.013291997456599564 0.048055683112321494 0.09968998092449673 -0.25305918234679936 0.005112306714076755; 0.06460275702939013 0.029816657090487753 0.00993888569682925 0.00993888569682925 0.00993888569682925 0.014908328545243876 0.16896105684609727 -0.3081054566017068]

    φ = [1.3, 1, -1.1, 0.5]
    μ = 0.5
    δ = 20
    ρ = 1 #0.1
    σ = 0
    present_time = 15

    truth = SigmoidalBranchingProcess(φ[2], φ[3], φ[1], φ[4], μ, δ, Γ, ρ, σ, type_space)

    num_treesets = 5
	num_trees_per_set = 52
    trees = rand_tree(truth, present_time, truth.type_space[5], num_treesets * num_trees_per_set; min_leaves=40)

    mkpath("out/")
    save_object("out/trees.jld2", trees)

	println("Sampling from posteriors...")
    dfs = Vector{DataFrame}(undef, num_treesets)

    Threads.@threads for i in 1:num_treesets
        treeset = trees[(i - 1) * num_trees_per_set + 1:i * num_trees_per_set]
        model = Model(treeset, Γ, type_space, ρ, σ, present_time)

        max_a_posteriori = optimize(model, MAP(), NelderMead())

        dfs[i] = sample(
            model,
            NUTS(adtype=AutoForwardDiff(chunksize=6)),
            1000,
            init_params=max_a_posteriori
        ) |> DataFrame

        dfs[i].run .= i
    end

    println("Exporting samples...")
    posterior_samples = vcat(dfs...)
    CSV.write("posterior-samples.csv", posterior_samples)

	println("Done!")
end

main()
