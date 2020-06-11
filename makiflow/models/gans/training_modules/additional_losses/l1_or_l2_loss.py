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


from makiflow.models.gans.main_modules import GeneratorDiscriminatorBasic

import tensorflow as tf


class L1orL2LossModuleGenerator(GeneratorDiscriminatorBasic):

    NOTIFY_BUILD_L1_SLASH_L2_LOSS = "L1/L2 loss was built"

    def __init__(self):
        self._l1_or_l2_loss_vars_are_ready = False

    def _prepare_training_vars(self):
        if not self._l1_or_l2_loss_vars_are_ready:
            self._use_l1 = False
            self._use_l1_or_l2_loss = False
            self._lambda = 1.0
            self._l1_or_l2_loss_is_built = False

            self._l1_or_l2_loss_vars_are_ready = True

    def is_use_l1_or_l2_loss(self) -> bool:
        """
        Return bool variable which shows whether it is being used l1/l2 or not.

        """
        if not self._l1_or_l2_loss_vars_are_ready:
            return self._l1_or_l2_loss_vars_are_ready

        return self._use_l1_or_l2_loss

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
        if not self._training_vars_are_ready:
            self._prepare_training_vars()
        # if `use_l1` is false, when l2 will be used
        self._use_l1 = use_l1
        self._lambda = scale
        self._use_l1_or_l2_loss = True

    def _build_l1_or_l2_loss(self):
        if not self._l1_or_l2_loss_is_built:
            if self._input_real_image is None:
                self._input_real_image = self._discriminator.get_inputs_maki_tensors()[0].get_data_tensor()

            if self._gen_product is None:
                # create output tensor from generator (in train set up)
                self._gen_product = self._return_training_graph_from_certain_output(
                    self._generator.get_outputs_maki_tensors()[0]
                )

            if self._use_l1:
                # build l1
                self._l1_or_l2_loss = tf.reduce_mean(tf.abs(self._gen_product - self._input_real_image)) * self._lambda
            else:
                # build l2
                self._l1_or_l2_loss = tf.reduce_mean(
                    tf.square(self._gen_product - self._input_real_image)
                ) * 0.5 * self._lambda
            print(L1orL2LossModuleGenerator.NOTIFY_BUILD_L1_SLASH_L2_LOSS)
            self._l1_or_l2_loss_is_built = True

        return self._l1_or_l2_loss

