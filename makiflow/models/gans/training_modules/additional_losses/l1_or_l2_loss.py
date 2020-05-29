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


from .basic_loss_module import BasicAdditionalLossModuleGenerator

import tensorflow as tf
import numpy as np

EPSILON = np.float32(1e-32)


class L1orL2LossModuleGenerator(BasicAdditionalLossModuleGenerator):

    def _build_l1_or_l2_loss(self):
        if self._use_l1:
            # build l1
            additional_loss = tf.reduce_mean(tf.abs(self._gen_product - self._input_real_image)) * self._lambda
        else:
            # build l2
            additional_loss = tf.reduce_mean(
                tf.square(self._gen_product - self._input_real_image)
            ) * 0.5 * self._lambda

        # add additional loss to final loss
        return additional_loss

