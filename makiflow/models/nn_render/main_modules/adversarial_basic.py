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


from .rendererbasic import NeuralRenderBasis
from makiflow.base.maki_entities import MakiTensor
from copy import copy


class AdversarialBasic(NeuralRenderBasis):

    def __init__(self, generator_model, discriminator_model, sampled_texture: MakiTensor, name):
        """
        Create adversarial module

        Parameters
        ----------
        generator_model : mf.models.nn_render.compile.NeuralRender
            Model of the Neural render
        discriminator_model : mf.models.nn_render.compile.NeuralRenderAdversarialTrainer
            Model for training neural render via adversarial loss
        sampled_texture : mf.MakiTensor
        name : str
        """
        gen_in_x = generator_model.get_inputs_maki_tensors()[0]
        gen_output_x = generator_model.get_outputs_maki_tensors()[0]

        self._generator_model = generator_model
        self._discriminator_model = discriminator_model

        disc_in_x = discriminator_model.get_inputs_maki_tensors()[0]
        disc_output_x = discriminator_model.get_outputs_maki_tensors()[0]

        connected_maki_tensors_output = self._connect_generator_disc_graph(gen_output_x, disc_in_x, disc_output_x)[0]

        self._use_l1 = False
        self._lambda = 1.0
        self._use_mask = False

        super().__init__(
            input_x=gen_in_x,
            output_x=connected_maki_tensors_output,
            sampled_texture=sampled_texture,
            name=name
        )

    def add_l1_or_l2_loss(self, use_l1=False, use_mask=False, scale=10.0):
        """
        Change additional loss and scale of it for model.

        Parameters
        ----------
        use_l1 : bool
            Add l1 loss for model, otherwise add l2 loss.
        use_mask : bool
            If true, in training mask will be used
        scale : float
            Scale of the additional loss.
        """
        self._use_l1 = use_l1
        self._lambda = scale
        self._use_mask = use_mask

    def _connect_generator_disc_graph(self, output_generator, input_discriminator, output_discriminator):
        """
        Connect two graph in one, which is used for training
        """
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

        # Create list of the layers names from discriminator
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

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------SETTING UP TRAINING FOR DISCRIMINATOR--------------------------------

    def _return_training_graph_from_certain_output(self, output: MakiTensor, train=True, name_layer_return=None):
        # Contains pairs {layer_name: tensor}, where `tensor` is output
        # tensor of layer called `layer_name`
        output_tensors = {}
        used = {}

        def create_tensor(from_):

            if used.get(from_.get_name()) is None:
                layer = from_.get_parent_layer()
                used[layer.get_name()] = True
                X = copy(from_.get_data_tensor())
                takes = []
                # Check if we at the beginning of the computational graph, i.e. InputLayer
                if len(from_.get_parent_tensor_names()) != 0:
                    for elem in from_.get_parent_tensors():
                        takes += [create_tensor(elem)]

                    if layer.get_name() in self._trainable_layers and train:
                        X = layer._training_forward(takes[0] if len(takes) == 1 else takes)
                    else:
                        X = layer._forward(takes[0] if len(takes) == 1 else takes)

                output_tensors[layer.get_name()] = X
                return X
            else:
                return output_tensors[from_.get_name()]

        training_outputs = create_tensor(output)
        if name_layer_return is not None:
            return output_tensors[name_layer_return]

        return training_outputs

    def get_input_shape(self):
        shape = self._inputs[0].get_shape()
        return shape
