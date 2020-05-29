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

import uuid
import os
import numpy as np

from evo.moo_interfaces import MOORobotInterface

class SoftbotRobot(MOORobotInterface):
    def __init__(self, phenotype, seq_num_gen, run_dir):
        self.run_dir = run_dir
        self.seq_num_gen = seq_num_gen
        self.seq_num = self.seq_num_gen()
        self.phenotype = phenotype
        self.evaluated_phenotype = self.phenotype.get_phenotype()
        self.morphology = None
        for (name, details) in self.phenotype.get_phenotype():
            if name == "material":
                self.morphology = details["state"]
        assert self.morphology is not None, "Morphology should not be None!"

        self.id = self.set_uuid()

        self.fitness = 0
        self.needs_eval = True

        self.age = 0

    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return "AFPO BOT: f: %.2f age: %d voxCnt: %d -- ID: %s" % (self.get_fitness(), self.get_age(),  np.sum(self.morphology > 0),  str(self.seq_num))


    def get_id(self):
        return self.seq_num

    def set_id(self, new_id):
        self.seq_num = new_id

    # Methods for MOORObotInterface class

    def iterate_generation(self):
        self.age += 1

    def needs_evaluation(self):
        return self.needs_eval

    def mutate(self):
        self.needs_eval = True
        self.phenotype.mutate()
        self.evaluated_phenotype = self.phenotype.get_phenotype()
        for (name, details) in self.phenotype.get_phenotype():
            if name == "material":
                self.morphology = details["state"]

        self.fitness = 0

        self.set_uuid()
        self.seq_num = self.seq_num_gen()

    def get_minimize_vals(self):
        return [self.get_age()]

    def get_maximize_vals(self):
        return [self.get_fitness()]

    def get_seq_num(self):
        return self.seq_num

    def get_fitness(self, test=False):
        return self.fitness

    def dominates_final_selection(self, other):
        """
        Used for printing generation summary statistics -- we only print the pareto frontier.
        :param other: The other SoftbotRobot to compare with this one
        :return: True if robot dominates the other robot, false otherwise.
        """
        return self.get_fitness() > other.get_fitness()

    # Methods for Work class
    def cpus_requested(self):
        return 1

    def compute_work(self, test=True, **kwargs):
        # convert numpy matrix of morphology to flattened file describing the morphology
        # for each voxel that is not air, write a line to the file in the format of
        # x,y,z|materialId

        morph_str = '\n'.join("%d,%d,%d|%d"%(*index,x) for index, x in np.ndenumerate(self.morphology) if x)
        with open("Robot_Morph_%.10d.txt"%self.get_seq_num(), "w") as f:
            f.write(morph_str)

        # run simulator to optimize morphology, compute and save fitness
        # self.fitness =
        # An example fitness function to minimize the number of voxels in a robot
        # self.fitness = -1 * np.sum(self.morphology > 0)


        # delete the morphology file.
        os.remove("Robot_Morph_%.10d.txt"%self.get_seq_num())

        raise NotImplementedError("Please implement robot evaluation. ")

    def write_letter(self):
        """
        When using ParallelPy with MPI for distributing simulations across multiple nodes,
        we only sync metadata about the simulation back to the evolutionary algorithm dispatch node.
        """
        return self.fitness

    def open_letter(self, letter):
        self.fitness = letter
        self.needs_eval = False
        return None

    def set_uuid(self):
        self.id = uuid.uuid1()
        return self.id

    def get_num_evaluations(self, test=False):
        return 1

    def get_age(self):
        return self.age

    def _flatten(self, l):
        ret = []
        for items in l:
            ret += items
        return ret

