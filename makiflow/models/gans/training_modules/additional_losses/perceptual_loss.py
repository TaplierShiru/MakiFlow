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

import numpy as np

EPSILON = np.float32(1e-32)


class PerceptualLossModuleGenerator(BasicAdditionalLossModuleGenerator):

    def _build_perceptual_loss(self):

        perceptual_loss = self._creation_per_loss(self._gen_product,
                                                  self._input_real_image,
                                                  self._session
        ) * self._scale_per_loss

        return perceptual_loss
