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


from .gansbasic import GANsBasic

class GeneratorDiscriminatorBasic(GANsBasic):

    def __init__(self, generator, discriminator, name='GeneratorDiscriminator'):
        gen_in_x = generator._inputs[0]
        gen_output_x = generator._outputs[0]

        self._generator = generator
        self._discriminator = discriminator

        disc_in_x = discriminator._inputs[0]
        disc_output_x = discriminator._outputs[0]

        connected_maki_tensors_output = self._connect_generator_disc_graph(gen_output_x, disc_in_x, disc_output_x)[0]

        super().__init__(input_s=gen_in_x, output=connected_maki_tensors_output, name=name)

    def _connect_generator_disc_graph(self, output_generator, input_discriminator, output_discriminator):
        stop_point = input_discriminator.get_name()
        outputs = [output_discriminator]
        
        used = {}
        layer_names = []
        def find_names(from_):
            if used.get(from_.get_name()) is None:
                layer = from_.get_parent_layer()
                used[layer.get_name()] = True
                # Check if we at the beginning of the computational graph, i.e. InputLayer
                if len(from_.get_parent_tensor_names()) != 0:
                    
                    for elem in from_.get_parent_tensor_names():
                        if stop_point == elem:
                            layer_names.append(from_.get_name())
                    
                    for elem in from_.get_parent_tensors():
                        find_names(elem)
                    
                        
        for output in outputs:
            find_names(output)  
        # Contains pairs {layer_name: tensor}, where `tensor` is output
        # tensor of layer called `layer_name`
        output_tensors = {}
        used = {}

        def create_tensor(from_):
            if used.get(from_.get_name()) is None:
                change = False

                layer = from_.get_parent_layer()
                used[layer.get_name()] = True
                X = from_
                takes = []
                if from_.get_name() in layer_names:
                    prev = [output_generator.get_name()]
                    change = True
                else:
                    prev = from_.get_parent_tensor_names()
                # Check if we at the beginning of the computational graph, i.e. InputLayer
                if len(prev) != 0:
                    if change:
                        takes += [create_tensor(output_generator)]
                    else:
                        for elem in from_.get_parent_tensors():
                            takes += [create_tensor(elem)]
                    
                    X = layer(takes[0] if len(takes) == 1 else takes)

                output_tensors[layer.get_name()] = X
                return X
            else:
                return output_tensors[from_.get_name()]
        
        training_outputs = []
        for output in outputs:
            training_outputs += [create_tensor(output)]

        return training_outputs

