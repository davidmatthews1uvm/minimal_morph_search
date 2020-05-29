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

import sys
import random
import numpy

from evo.afpomoo import AFPOMoo
from softbot_robot import SoftbotRobot
from utils import StructureGenotype, StructurePhenotype, get_seq_num


# To change the number of materials,
# change the value of NUM_MATERIALS in utils.py

# To change the size of the robot,
# change the value of IND_SIZE in utils.py

# To change if we allow robots to have non-unique morphologies,
# change the value of FORCE_MORPH_ONCE in utils.py
# by default is set to True meaning every robot will have a unique morphology


POP_SIZE = 21 # how large of a population are we using?
GENS = 200 # how many generations are we optimizing for?

printing = True

if __name__ == '__main__':
    assert len(sys.argv) >= 2, "please run as python job.py seed"
    seed = int(sys.argv[1])

    numpy.random.seed(seed)
    random.seed(seed)

    # Setup evo run
    def get_phenotype():
        new_phenotype = StructurePhenotype(StructureGenotype)
        return new_phenotype

    def robot_factory():
        phenotype = get_phenotype()
        return SoftbotRobot(phenotype, get_seq_num, "run_%d" % seed)

    afpo_alg = AFPOMoo(robot_factory, pop_size=POP_SIZE)

    # do each generation.
    for generation in range(GENS):
        if printing:
            print("generation %d" % (generation))

        dom_data = afpo_alg.generation()

        if printing:
            print("%d individuals are dominating" % (dom_data[0],))
            dom_inds = sorted(dom_data[1], key= lambda x: x.get_fitness(), reverse=False)
            print('\n'.join([str(d) for d in dom_inds]))

        best_fit, best_robot = afpo_alg.get_best()

