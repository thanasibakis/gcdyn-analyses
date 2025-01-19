# Simulation studies without a carrying capacity

This directory hosts simulation studies that do not impose a carrying capacity on the trees they generate.

## Descriptions of each study

`approximate-likelihood`
- Generates trees under our branching process; performs inference with an approximation to the likelihood
- See Section 3.2.3 of the paper

`incorrect-sampling-prob`
- Generates trees under our branching process; performs inference with the leaf sampling probability fixed to an incorrect value
- See Section 3.2.4 of the paper

`no-misspecification`
- Generates trees under our branching process; performs inference with the matching likelihood
- See Section 3.2.2 of the paper

`unconditioned-likelihood`
- Generates trees under a single-type, constant-rate branching process; performs inference with and without conditioning on non-extinction
- See Section 3.2.1 of the paper

## Instructions for running

To run each simulation study, enter the directory and run:

```shell
julia --project --threads=5 main.jl
```

Posterior samples will be saved in the `out/` subdirectory.
