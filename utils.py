# Copyright 2020 David Matthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from queue import Queue

import numpy as np
from evosorocore.Genome import Genotype, Phenotype, make_material_tree
from evosorocore.Networks import CPPN

IND_SIZE = (8,8,7)
MIN_PERCENT_FULL = 0.5

# build a full binary tree that is mostly balanced with NUM_MATERIALS + 1 leaf nodes.
NUM_MATERIALS = 10 # at a minimum we need 1 material.)
assert NUM_MATERIALS >= 1, "NUM_MATERIALS must be >= 1"

class Node(object):
    def __init__(self, id, isLeaf):
        self.id = id
        self.isLeaf = isLeaf
        self.parentId = None
        self.childA = None
        self.childB = None

    def __str__(self):
        if self.isLeaf:
            return ""
        else:
            return "Internal node %d with parent %s and children %d %d" %(self.id, str(self.parentId), self.childA.id, self.childB.id)

all_nodes = [Node(id, True) for id in range(NUM_MATERIALS + 1)]

currentNodeIdx = NUM_MATERIALS + 1

nodes_needing_parents = Queue()

for node in all_nodes:
    nodes_needing_parents.put(node)

while (nodes_needing_parents.qsize() >= 2):
    child_a  = nodes_needing_parents.get()
    child_b  = nodes_needing_parents.get()
    child_a.parentId = str(currentNodeIdx)
    child_b.parentId = str(currentNodeIdx)
    parent = Node(str(currentNodeIdx), False)
    parent.childA = child_a
    parent.childB = child_b
    nodes_needing_parents.put(parent)
    all_nodes.append(parent)
    currentNodeIdx += 1

NODE_NAMES = [n.id for n in reversed(all_nodes[NUM_MATERIALS + 1:])]

FORCE_MORPH_ONCE = False
MORPHOLOGIES_SEEN_BEFORE = {}

robot_seq_number = 0
def get_seq_num():
    global robot_seq_number
    robot_seq_number += 1
    return robot_seq_number


class StructureGenotype(Genotype):
    def __init__(self):
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        self.add_network(CPPN(output_node_names=NODE_NAMES))

        self.to_phenotype_mapping.add_map(name="material", tag="<Data>", func=make_material_tree,
                                          dependency_order=NODE_NAMES, output_type=int)

        for node in reversed(all_nodes[NUM_MATERIALS + 1:]):
            l_child = str(node.childA.id) if node.childA.isLeaf else None
            r_child = str(node.childB.id) if node.childB.isLeaf else None
            parent = node.parentId

            if parent is None:
                self.to_phenotype_mapping.add_output_dependency(name=node.id, dependency_name=None, requirement=None,
                                                                material_if_true=l_child, material_if_false=r_child)
            else:
                self.to_phenotype_mapping.add_output_dependency(name=node.id, dependency_name=parent, requirement=False,
                                                material_if_true=l_child, material_if_false=r_child)

class StructurePhenotype(Phenotype):
    def is_valid(self, min_percent_full=MIN_PERCENT_FULL):
        for name, details in self.genotype.to_phenotype_mapping.items():
            if np.isnan(details["state"]).any():  # no value should be NAN.
                return False

            if name == "material":  # check material states.
                state = details["state"]

                if FORCE_MORPH_ONCE and tuple(state.flatten()) in MORPHOLOGIES_SEEN_BEFORE:
                    return False

                # for robot to not be entirely empty space.
                num_vox = np.sum(state > 0)
                if num_vox < np.product(self.genotype.orig_size_xyz) * min_percent_full:  # must have enough structure
                    return False

                if FORCE_MORPH_ONCE:
                    MORPHOLOGIES_SEEN_BEFORE[tuple(state.flatten())] = 1

        return True

