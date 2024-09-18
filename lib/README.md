File dependencies:
- `Ab-CGGnaive_DMS/results/final_variant_scores/final_variant_scores.csv`
- `gcreplay/nextflow/results/archive/2024-06-23-beast-15-day/beast/`
- `gcreplay/analysis/output/10x/data.csv`

Package dependencies:
- `pip install -e ./gcdyn/`
- `julia --project -e 'using Pkg; Pkg.develop(path="lib/gcdyn.jl")'`
