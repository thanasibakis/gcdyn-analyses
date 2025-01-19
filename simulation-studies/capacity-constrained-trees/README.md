# Simulation studies with a carrying capacity

This directory hosts simulation studies that impose a carrying capacity on the trees they generate.

## Descriptions of each study

`no-other-misspecification`
- Generates trees under a hard carrying capacity and a mutation process on the discrete type space; performs inference with our branching process
- See Section 3.2.5 of the paper

`no-other-misspecification-soft-capacity`
- Generates trees under a soft carrying capacity and a mutation process on the discrete type space; performs inference with our branching process
- See Section 3.2.6 of the paper

`sequence-level-mutation`
- Generates trees under a hard carrying capacity and a genetic sequence-level mutation process; performs inference with our branching process
- See Section 3.2.7 of the paper

## Instructions for running

To run each simulation study, enter the directory and run:

```shell
python generate-trees.py
julia --project parse-trees.jl
julia --project --threads=5 main.jl
```

Posterior samples will be saved in the `out/` subdirectory.