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
from .l1_regularization import L1
from abc import ABC


# noinspection PyTypeChecker
class L2(L1, ABC):
    def _init(self):
        super()._init()
        # Setup L2 regularization
        self._uses_l2_regularization = False
        self._l2_reg_loss_is_build = False
        self._l2_regularized_layers = {}
        for layer_name in self._trainable_layers:
            self._l2_regularized_layers[layer_name] = None

    def set_l2_reg(self, layers):
        """
        Enables L2 regularization while training and allows to set different
        decays to different weights.

        Parameters
        ----------
        layers : list
            Contains tuples (layer_name, decay) where decay is float number(set it to
            None if you want to turn off regularization on this weight).
        """
        # noinspection PyAttributeOutsideInit
        self._uses_l2_regularization = True

        for layer_name, decay in layers:
            self._l2_regularized_layers[layer_name] = decay

    def set_common_l2_weight_decay(self, decay=1e-6):
        """
        Sets `decay` for all trainable weights (that can be regularized).

        Parameters
        ----------
        decay : float
            Decay for all the regularized weights.
        """
        # noinspection PyAttributeOutsideInit
        self._uses_l2_regularization = True

        for layer_name in self._l2_regularized_layers:
            self._l2_regularized_layers[layer_name] = decay

    def __build_l2_loss(self):
        self._l2_reg_loss = tf.constant(0.0)
        for layer_name in self._l2_regularized_layers:
            decay = self._l2_regularized_layers[layer_name]
            if decay is not None:
                layer = self._graph_tensors[layer_name].get_parent_layer()
                params = layer.get_params_regularize()
                for param in params:
                    self._l2_reg_loss += tf.nn.l2_loss(param) * tf.constant(decay)

        self._l2_reg_loss_is_build = True

    def _build_final_loss(self, training_loss):
        if self._uses_l2_regularization:
            self.__build_l2_loss()
            training_loss += self._l2_reg_loss

        return super()._build_final_loss(training_loss)

    def get_l2_regularization_loss(self):
        assert self._l2_reg_loss_is_build, 'The loss has not been built yet. Please compile the model'
        return self._l2_reg_loss
