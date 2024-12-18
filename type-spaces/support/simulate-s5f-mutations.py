#!/usr/bin/env python3

## `simulate-s5f-mutations`
#
# Reads a list of sequences (separated by newlines) from stdin or a file passed as an argument, and runs a mutation process on each sequence for a fixed amount of time.
# On one line per sequence, the details of each mutation are printed in the form "fromKD:duration:toKD".
# Semicolons separate each mutation.
#
# Requirements:
# - Tested on Python 3.9.
# - Requires that `lib/gcdyn` is installed from the repository root directory.


import fileinput
import warnings
from pathlib import Path

from experiments import replay
from gcdyn import bdms, gpmap, mutators, poisson

dms = replay.dms(Path(__file__).parent / "final_variant_scores.csv")
gp_map = gpmap.AdditiveGPMap(
    dms["affinity"], nonsense_phenotype=dms["affinity"].min().min()
)

mutator = mutators.SequencePhenotypeMutator(
    mutators.ContextMutator(
        mutability=replay.mutability(), substitution=replay.substitution()
    ),
    gp_map,
)

with warnings.catch_warnings():
    # Safe to ignore divide by zero warnings
    warnings.simplefilter("ignore")

    for line in fileinput.input():
        sequence = line.strip()

        node = bdms.TreeNode()
        node.sequence = sequence
        node.x = gp_map(sequence)
        node.chain_2_start_idx = replay.CHAIN_2_START_IDX

        node.evolve(
            t=200,
            birth_response=poisson.ConstantResponse(0),
            death_response=poisson.ConstantResponse(0),
            mutation_response=poisson.ConstantResponse(1),
            mutator=mutator,
            verbose=False,
        )

        transitions = []

        while len(node.children) > 0:
            node = node.children[0]
            transitions.append(f"{node.up.x}:{node.t - node.up.t}:{node.x}")

        print(";".join(transitions))
