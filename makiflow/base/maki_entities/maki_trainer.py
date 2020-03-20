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
from makiflow.base.maki_entities.maki_model import MakiModel
import tensorflow as tf
from copy import copy


class MakiTrainer(MakiModel, ABC):
    # Provides API for training the model.
    def __init__(self, graph_tensors: dict, outputs: list, inputs: list):
        self._set_for_training = False
        super().__init__(graph_tensors, outputs, inputs)

    def _setup_for_training(self):
        self._set_for_training = True

        # Collect all the layers names since all of them are trainable from
        # the beginning.
        self._trainable_layers = []
        for layer_name in self._graph_tensors:
            self._trainable_layers.append(layer_name)
        self._trainable_vars = []
        self._collect_train_params()

        # Setup L2 regularization
        self._uses_l2_regularization = False
        self._l2_reg_loss_is_build = False
        self._l2_regularized_layers = {}
        for layer_name in self._trainable_layers:
            self._l2_regularized_layers[layer_name] = 1e-6  # This value seems to be proper as a default

        self._uses_l1_regularization = False
        self._l1_reg_loss_is_build = False
        self._l1_regularized_layers = {}
        for layer_name in self._trainable_layers:
            self._l1_regularized_layers[layer_name] = 1e-6  # This value seems to be proper as a default

    def set_layers_trainable(self, layers):
        """

        Parameters
        ----------
        layers: list
            List of tuples: (layer_name, bool).
            Example: to make layer called `conv1` untrainable use
            the following tuple (`conv1`, False), if you want to make it trainable
            use the following tuple (`conv1`, True).
        """
        if not self._set_for_training:
            self._setup_for_training()
        for layer_name, is_trainable in layers:
            if is_trainable and layer_name not in self._trainable_layers:
                self._trainable_layers.append(layer_name)
            elif not is_trainable and layer_name in self._trainable_layers:
                self._trainable_layers.remove(layer_name)
        # It will collect the trainable parameters and rebuild the graph.
        # Graph rebuild is needed since some of the layers behave differently in
        # training mode.
        self._collect_train_params()

    def _collect_train_params(self):
        self._trainable_vars.clear()
        for layer_name in self._trainable_layers:
            layer = self._graph_tensors[layer_name].get_parent_layer()
            self._trainable_vars += layer.get_params()
        # Create graph or refresh it
        self._build_training_graph()

    # L2 REGULARIZATION
    def set_l2_reg(self, layers):
        """
        Enables L2 regularization while training and allows to set different
        decays to different weights.
        WARNING! It is assumed that `set_layers_trainable` method won't
        be used anymore.

        Parameters
        ----------
        layers : list
            Contains tuples (layer_name, decay) where decay is float number(set it to
            None if you want to turn off regularization on this weight).
        """
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l2_regularization = True

        for layer_name, decay in layers:
            self._l2_regularized_layers[layer_name] = decay

    def set_common_l2_weight_decay(self, decay=1e-6):
        """
        Enables L2 regularization while training.
        `decay` will be set as decay for each regularized weight.
        If you haven't used `set_l2_reg` method and did not turn off
        the regularization on certain layers, the regularization will be
        set on all the trainable layers.

        Parameters
        ----------
        decay : float
            Decay for all the regularized weights.
        """
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l2_regularization = True

        for layer_name in self._l2_regularized_layers:
            if self._l2_regularized_layers[layer_name] is not None:
                self._l2_regularized_layers[layer_name] = decay

    def _build_l2_loss(self):
        self._l2_reg_loss = tf.constant(0.0)
        for layer_name in self._l2_regularized_layers:
            decay = self._l2_regularized_layers[layer_name]
            if decay is not None:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                params = layer.get_params()
                for param in params:
                    self._l2_reg_loss += tf.nn.l2_loss(param) * tf.constant(decay)

        self._l2_reg_loss_is_build = True

    # L1 REGULARIZATION

    def set_l1_reg(self, layers):
        """
        Enables L2 regularization while training and allows to set different
        decays to different weights.
        WARNING! It is assumed that `set_layers_trainable` method won't
        be used anymore.

        Parameters
        ----------
        layers : list
            Contains tuples (layer_name, decay) where decay is float number(set it to
            None if you want to turn off regularization on this weight).
        """
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l1_regularization = True

        for layer_name, decay in layers:
            self._l1_regularized_layers[layer_name] = decay

    def set_common_l1_weight_decay(self, decay=1e-6):
        """
        Enables L2 regularization while training.
        `decay` will be set as decay for each regularized weight.
        If you haven't used `set_l1_reg` method and did not turn off
        the regularization on certain layers, the regularization will be
        set on all the trainable layers.

        Parameters
        ----------
        decay : float
            Decay for all the regularized weights.
        """
        if not self._set_for_training:
            self._setup_for_training()

        self._uses_l1_regularization = True

        for layer_name in self._l1_regularized_layers:
            if self._l1_regularized_layers[layer_name] is not None:
                self._l1_regularized_layers[layer_name] = decay

    def _build_l1_loss(self):
        self._l1_reg_loss = tf.constant(0.0)
        for layer_name in self._l1_regularized_layers:
            decay = self._l1_regularized_layers[layer_name]
            if decay is not None:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                params = layer.get_params()
                for param in params:
                    self._l1_reg_loss += tf.abs(tf.reduce_sum(param)) * tf.constant(decay)

        self._l1_reg_loss_is_build = True

    def _build_final_loss(self, custom_loss):
        # Adds regularization terms to the actual loss.
        if self._uses_l1_regularization:
            if not self._l1_reg_loss_is_build:
                self._build_l1_loss()
            custom_loss += self._l1_reg_loss

        if self._uses_l2_regularization:
            if not self._l2_reg_loss_is_build:
                self._build_l2_loss()
            custom_loss += self._l2_reg_loss

        return custom_loss

    def _build_training_graph(self):
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
                if from_.get_parent_tensor_names() is not None:
                    for elem in from_.get_parent_tensors():
                        takes += [create_tensor(elem)]

                    if layer.get_name() in self._trainable_layers:
                        X = layer._training_forward(takes[0] if len(takes) == 1 else takes)
                    else:
                        X = layer._forward(takes[0] if len(takes) == 1 else takes)

                output_tensors[layer.get_name()] = X
                return X
            else:
                return output_tensors[from_.get_name()]

        self._training_outputs = []
        for output in self._outputs:
            self._training_outputs += [create_tensor(output)]