# Bayesian inference of antibody evolutionary dynamics

A collection of analyses for the paper, *Bayesian inference of antibody evolutionary dynamics using multitype branching processes*.

There are also accompanying software packages which implement branching process simulators and likelihoods:
- [thanasibakis/gcdyn.jl](https://github.com/thanasibakis/gcdyn.jl)
- [matsengrp/gcdyn](https://github.com/matsengrp/gcdyn)

## Getting started

0. Please ensure you have a working Python and [Julia](https://julialang.org/downloads/) installation.
(This codebase was tested with Python 3.12 and Julia 1.10.)

1. Set up a Python environment for the `gcdyn` dependency.

```shell
python -m venv .venv/
echo "*" > .venv/.gitignore
source .venv/bin/activate

pip install -e lib/gcdyn
```

2. Set up the Julia environment.

```shell
julia  --project -e "import Pkg; Pkg.instantiate()"
```

## Repository structure

Visit the following directories for specific instructions on how to run each analysis:

`data-analysis/`
- Analysis of our real experimental dataset
- See Section 3.3 of the paper

`simulation-studies/`
- A collection of simulation studies to test our method under various model misspecifications
- See Section 3.2 of the paper

`type-spaces/`
- Code to compute our type space and type change rate matrix
- See Section 3.1 of the paper
