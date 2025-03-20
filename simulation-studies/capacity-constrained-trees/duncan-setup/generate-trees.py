#!/usr/bin/env python3

import json
import os
import pickle


def export_tree(tree):
    return {
        "affinity": tree.x,
        "time": tree.t,
        "event": tree.event if tree.event else "root",
        "children": [export_tree(child) for child in tree.children],
    }


os.makedirs("out/trees/", exist_ok=True)

json_trees = []

for path in os.listdir("data/"):
    with open(f"data/{path}", "rb") as f:
        tree = pickle.load(f)["tree"]
        json_trees.append(export_tree(tree))

with open("out/trees/trees.json", "w") as f:
    json.dump(json_trees, f)
