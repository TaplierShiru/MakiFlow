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

from makiflow.generators.pipeline.map_method import MapMethod
import tensorflow as tf

from .generators import GANsPathGenerator
from .pipeline_base import GenLayer


class GANsCreationIterator():

    def __init__(self, prefetch_size, batch_size, using_images_for_gen, path_generator: GANsPathGenerator,
                 map_operation: MapMethod, num_parallel_calls=None
    ):
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        self.iterator = self.build_iterator(path_generator, map_operation,
                                            num_parallel_calls, using_images_for_gen
        )

    def build_iterator(self, gen: GANsPathGenerator, map_operation: MapMethod,
                       num_parallel_calls, using_images_for_gen):
        if using_images_for_gen:
            dataset = tf.data.Dataset.from_generator(
                gen.next_element,
                output_types={
                    GANsPathGenerator.IMAGE: tf.string,
                    GANsPathGenerator.GEN_INPUT: tf.string
                }
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                gen.next_element,
                output_types={
                    GANsPathGenerator.IMAGE: tf.string
                }
            )
        dataset = dataset.map(map_func=map_operation.load_data, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.prefetch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def get_iterator(self):
        return self.iterator


class InputGenLayer(GenLayer):
    def __init__(
            self, iterator: GANsCreationIterator, name, input_tensor_name: str
    ):
        self.iterator = iterator
        self.input_tensor_name = input_tensor_name
        super().__init__(
            name=name,
            input_tensor=self.iterator.get_iterator()[input_tensor_name]
        )

    def get_iterator(self):
        return self.iterator.get_iterator()[self.input_tensor_name]
