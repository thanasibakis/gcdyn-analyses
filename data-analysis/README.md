# Analysis of experimental germinal center data

Instructions for running:

1. Load BEAST trees into `./data/raw/`
	- This directory should contain one subdirectory for each germinal center, with each containing the `.history.trees` file
	- While the full set of data from the replay experiment was not published at the time of this analysis, a sample of the trees is given here in `sample-beast-trees.tar.xz`
2. Run `julia --project parse-beast.jl` to process these into a format usable for the Julia analysis code
3. (Optional) run `julia --project visualize-trees.jl` to generate plots of a sample of the trees for each germinal center
	- Plots will be saved to `out/tree-visualizations/`
4. Run `SEED=1 julia --project main.jl` to run MCMC with a random seed of 1
	- You are welcome to change the seed
	- The posterior samples will be saved to `out/seed-$SEED/`
5. (Optional) run `SEED=1 julia --project posterior-predictive-check.jl` to collect affinities from trees drawn from the posterior predictive distribution of the MCMC run.
	- You are welcome to change the seed
	- The affinities will be saved to `out/seed-$SEED/`