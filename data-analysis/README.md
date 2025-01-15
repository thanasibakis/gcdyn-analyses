# Analysis of experimental germinal center data

Instructions for running:

1. Load BEAST trees from **TODO** into `./data/raw/`
	- This directory should contain one subdirectory for each germinal center, with each containing the `.history.trees` file
2. Run `julia --project parse-beast.jl` to process these into a format usable for the Julia analysis code
3. (Optional) run `julia --project visualize-trees.jl` to generate plots of a sample of the trees for each germinal center
	- Plots will be saved to `out/tree-visualizations/`
4. Run `SEED=1 julia --project main.jl` to run MCMC with a random seed of 1
	- You are welcome to change the seed
	- The posterior samples will be saved to `out/seed-$SEED/`