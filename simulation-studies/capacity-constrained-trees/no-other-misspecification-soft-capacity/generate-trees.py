#!/usr/bin/env python3

import json
import os

import numpy as np
from gcdyn.bdms import TreeError, TreeNode
from gcdyn.mutators import DiscreteMutator
from gcdyn.poisson import (
    ConstantResponse,
    SigmoidResponse,
)
from numpy.random import default_rng

birth_response = SigmoidResponse(1.5, -0.1, 2.5, 0.6)
death_response = ConstantResponse(1.0)
mutation_response = ConstantResponse(1.0)
present_time = 15
survivor_sampling_prob = 0.1


type_space = [
    -2.4270176906430416,
    -1.4399117849363843,
    -0.6588015552361666,
    -0.13202968692343608,
    0.08165101396850624,
    0.7981793588605735,
    1.3526378568771724,
    2.1758707012574643,
]
Γ = np.array(
    [
        [
            -0.004619127041534006,
            0.0033511313830736906,
            0.0004528555923072555,
            0.00027171335538435327,
            0.0003622844738458044,
            9.05711184614511e-5,
            9.05711184614511e-5,
            0.0,
        ],
        [
            0.2020753168875446,
            -0.2209145269997298,
            0.013881523240557529,
            0.0031729195978417207,
            0.0009915373743255376,
            0.00019830747486510755,
            0.0005949224245953227,
            0.0,
        ],
        [
            0.1102975726069481,
            0.15072536814415619,
            -0.28848747576279454,
            0.02263077684963278,
            0.004614041881963965,
            0.00021971628009352211,
            0.0,
            0.0,
        ],
        [
            0.0921587483881802,
            0.051134186307591235,
            0.0984589489934135,
            -0.2857067716326731,
            0.03618952440680526,
            0.00776536353668291,
            0.0,
            0.0,
        ],
        [
            0.0865376168167631,
            0.02461627223233437,
            0.06534732061676392,
            0.13285173725388705,
            -0.3320658991341189,
            0.01827187217245438,
            0.004441080041915994,
            0.0,
        ],
        [
            0.07206014315123183,
            0.021894379777512914,
            0.015304809164863396,
            0.048677795816023856,
            0.10096922712930713,
            -0.2835641031378857,
            0.024657748098946584,
            0.0,
        ],
        [
            0.05828029654047501,
            0.020704842192010856,
            0.00792407540681897,
            0.013291997456599564,
            0.048055683112321494,
            0.09968998092449673,
            -0.25305918234679936,
            0.005112306714076755,
        ],
        [
            0.06460275702939013,
            0.029816657090487753,
            0.00993888569682925,
            0.00993888569682925,
            0.00993888569682925,
            0.014908328545243876,
            0.16896105684609727,
            -0.3081054566017068,
        ],
    ]
)
transition_matrix = Γ
transition_matrix[np.diag_indices_from(transition_matrix)] = 0
transition_matrix /= np.sum(transition_matrix, axis=1)[:, np.newaxis]

mutator = DiscreteMutator(state_space=type_space, transition_matrix=transition_matrix)

num_trees = 5 * 52
rng = default_rng(1)


def generate_tree(_tries=1):
    try:
        root = TreeNode()
        root.x = type_space[4]

        root.evolve(
            t=present_time,
            birth_response=birth_response,
            death_response=death_response,
            mutation_response=mutation_response,
            mutator=mutator,
            verbose=False,
            capacity=1000,
            capacity_method="birth",
            min_survivors=0,
            seed=rng,
        )

        root.sample_survivors(p=survivor_sampling_prob)

        survivors = filter(
            lambda node: node.event == "sampling",
            root.get_leaves(),
        )

        if len(list(survivors)) == 0:
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


# Export full, unpruned trees to understand population size over time
trees = [generate_tree() for _ in range(num_trees)]
json_trees = [export_tree(tree) for tree in trees]

os.makedirs("out/trees/", exist_ok=True)

with open("out/trees/trees-unpruned.json", "w") as f:
    json.dump(json_trees, f)

# Export pruned trees to use for inference
for tree in trees:
    tree.prune()

json_trees = [export_tree(tree) for tree in trees]

with open("out/trees/trees.json", "w") as f:
    json.dump(json_trees, f)
