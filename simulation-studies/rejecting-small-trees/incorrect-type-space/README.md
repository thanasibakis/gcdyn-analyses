First run `python3 generate-trees.jl`, then `julia --project parse-trees.jl`.
This will create `trees.json` and `trees.jld2`, respectively.
Then, run `main.jl` in either subdirectory, which depends on `trees.jld2`.