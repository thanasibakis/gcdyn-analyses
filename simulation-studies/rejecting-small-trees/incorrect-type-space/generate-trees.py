#!/usr/bin/env python3

import json

from experiments import replay
from gcdyn.bdms import TreeError, TreeNode
from gcdyn.gpmap import AdditiveGPMap
from gcdyn.mutators import ContextMutator, SequencePhenotypeMutator
from gcdyn.poisson import (
    ConstantResponse,
    SequenceContextMutationResponse,
    SigmoidResponse,
)
from numpy.random import default_rng

naive_sequence = "GAGGTGCAGCTTCAGGAGTCAGGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGTGCAAGGGACTTCGATGTCTGGGGCGCAGGGACCACGGTCACCGTCTCCTCAGACATTGTGATGACTCAGTCTCAAAAATTCATGTCCACATCAGTAGGAGACAGGGTCAGCGTCACCTGCAAGGCCAGTCAGAATGTGGGTACTAATGTAGCCTGGTATCAACAGAAACCAGGGCAATCTCCTAAAGCACTGATTTACTCGGCATCCTACAGGTACAGTGGAGTCCCTGATCGCTTCACAGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAATGTGCAGTCTGAAGACTTGGCAGAGTATTTCTGTCAGCAATATAACAGCTATCCTCTCACGTTCGGCTCGGGGACTAAGCTAGAAATAAAA"
birth_response = SigmoidResponse(1.0, -1.1, 1.3, 0.5)
death_response = ConstantResponse(5.5)
mutation_response = SequenceContextMutationResponse(
    replay.mutability(), mutation_intensity=1.75
)
present_time = 15
survivor_sampling_prob = 0.1

dms = replay.dms("final_variant_scores.csv")

gp_map = AdditiveGPMap(
    dms["affinity"], nonsense_phenotype=dms["affinity"].min(axis=None)
)
mutator = SequencePhenotypeMutator(
    ContextMutator(mutability=replay.mutability(), substitution=replay.substitution()),
    gp_map,
)

num_trees = 5 * 52
rng = default_rng(1)


def generate_tree(_tries=1):
    try:
        root = TreeNode()
        root.sequence = naive_sequence
        root.x = gp_map(naive_sequence)
        root.chain_2_start_idx = replay.CHAIN_2_START_IDX

        root.evolve(
            t=present_time,
            birth_response=birth_response,
            death_response=death_response,
            mutation_response=mutation_response,
            mutator=mutator,
            verbose=False,
            min_survivors=0,
            seed=rng,
        )

        root.sample_survivors(p=survivor_sampling_prob)
        root.prune()

        survivors = filter(
            lambda node: node.event == "sampling",
            root.get_leaves(),
        )

        if len(list(survivors)) < 40:
            return generate_tree(_tries + 1)

        print(f"Generated tree after {_tries} tries")

        return root

    except TreeError:
        # Tree went extinct, try again
        return generate_tree(_tries + 1)


def export_tree(tree):
    return {
        "affinity": tree.x,
        "time": tree.t,
        "event": tree.event if tree.event else "root",
        "children": [export_tree(child) for child in tree.children],
    }


trees = [generate_tree() for _ in range(num_trees)]
json_trees = [export_tree(tree) for tree in trees]

with open("trees.json", "w") as f:
    json.dump(json_trees, f)
