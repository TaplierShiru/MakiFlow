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


from .l1_or_l2_loss import L1orL2LossModuleGenerator
from .perceptual_loss import PerceptualLossModuleGenerator

import tensorflow as tf
from makiflow.models.common.utils import new_optimizer_used, loss_is_built
import numpy as np

EPSILON = np.float32(1e-32)


class BasicTrainingModule(L1orL2LossModuleGenerator,
                          PerceptualLossModuleGenerator):

    def _build_additional_losses(self, total_loss):
        if self.is_use_l1():
            total_loss += self._build_l1_or_l2_loss()

        if self.is_use_perceptual_loss():
            total_loss += self._build_perceptual_loss()

        return total_loss

