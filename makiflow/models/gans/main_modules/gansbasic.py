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


import tensorflow as tf
from makiflow.base.maki_entities import MakiTensor
from makiflow.base.maki_entities import MakiCore
from copy import copy

class GANsBasic(MakiCore):

    NAME = 'name'
    INPUT_S = 'input_s'
    OUTPUT = 'output'

    @staticmethod
    def from_json(path_to_model):
        # TODO
        pass

    def __init__(self, input_s, output, name):
        self.name = str(name)
        graph_tensors = output.get_previous_tensors()
        graph_tensors.update(output.get_self_pair())
        super().__init__(graph_tensors, outputs=[output], inputs=[input_s])
        self._inference_out = self._output_data_tensors[0]

        self._generator_is_set = False
        self._training_vars_are_ready = False
        self._binary_ce_loss_is_built = False

        self._feature_matching_loss_is_built = False
        self._feature_loss = False
        self._feature_scale = 1.0
        self._use_feature_loss_only = False

        self._wasserstein_is_built = False
        
        self._use_l1 = None
        self._lambda = None

    def predict(self, x):
        return self._session.run(
            self._output_data_tensors[0],
            feed_dict={self._input_data_tensors[0]: x}
        )

    def is_use_l1(self) -> bool:
        """
        Return bool variable which shows whether it is being used l1/l2 or not.
        """
        return self._use_l1 if self._use_l1 is not None else False

    def _get_model_info(self):
        return {
            GANsBasic.NAME: self.name,
            GANsBasic.INPUT_S: self._inputs[0].get_name(),
            GANsBasic.OUTPUT: self._outputs[0].get_name()
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------SETTING UP TRAINING-------------------------------------

    def get_layers_names(self) -> dict:
        """
        Returns
        -------
        dict of MakiTensors
            All the MakiTensors that appear earlier in the computational graph.
            The dictionary contains pairs: { name of the tensor: MakiTensor }.
        """
        return self._graph_tensors

    def get_logits_shape(self):
        return self._outputs[0].get_shape()

    def get_outputs_maki_tensors(self) -> list:
        return self._outputs

    def get_input_shape(self):
        shape = self._inputs[0].get_shape()
        return shape

    def get_inputs_maki_tensors(self) -> list:
        return self._inputs

    # ------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------ADDITIONAL TOOLS FOR TRAINING-----------------------------------

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

    def add_l1_or_l2_loss(self, use_l1=True, scale=10.0):
        """
        Add additional loss for model.

        Parameters
        ----------
        use_l1 : bool
            Add l1 loss for model, otherwise add l2 loss.
        scale : float
            Scale of the additional loss.
        """
        # if `use_l1` is false, when l2 will be used
        self._use_l1 = use_l1
        self._lambda = scale

