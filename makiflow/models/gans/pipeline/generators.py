# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from abc import ABC
from glob import glob
import os
from sklearn.utils import shuffle

from .pipeline_base import PathGenerator


class GANsPathGenerator(PathGenerator, ABC):
    IMAGE = 'image'
    GEN_INPUT = 'gen_input'


class CyclicGeneratorGANs(GANsPathGenerator):
    def __init__(self, path_images, path_gen_inputs, type_image='png', type_gen_inputs='png'):
        """
        Generator for pipeline, which gives next element in cycle order
        Parameters
        ----------
        path_images : str
            Path to the gen_inputs folder. Example: /home/mnt/data/batch_1/gen_inputs
        path_gen_inputs : str
            Path to the images folder. Example: /home/mnt/data/batch_1/images.
            Set it to None, if your generator training on some noise.
        """
        self.images = glob(os.path.join(path_images, f'*.{type_image}'))
        if path_gen_inputs is not None:
            self.gen_inputs = glob(os.path.join(path_gen_inputs, f'*.{type_gen_inputs}'))
        else:
            self.gen_inputs = path_gen_inputs

    def next_element(self):
        index = 0
        if self.gen_inputs is not None:
            while True:
                if index >= len(self.images):
                    self.images, self.gen_inputs = shuffle(self.images, self.gen_inputs)
                    index = 0

                el = {
                    GANsPathGenerator.IMAGE: self.images[index],
                    GANsPathGenerator.GEN_INPUT: self.gen_inputs[index]
                }
                index += 1

                yield el
        else:
            while True:
                if index >= len(self.images):
                    self.images = shuffle(self.images)
                    index = 0

                el = {
                    GANsPathGenerator.IMAGE: self.images[index]
                }
                index += 1
                yield el
