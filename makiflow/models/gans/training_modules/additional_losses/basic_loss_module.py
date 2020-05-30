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
import numpy as np


class BasicAdditionalLossModuleGenerator(GeneratorDiscriminatorBasic):

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()
        # prepare inputs and outputs for l1 or l2 if it need
        if self.is_use_l1() or self.is_use_perceptual_loss():
            if self._input_real_image is None:
                self._input_real_image = self._discriminator.get_inputs_maki_tensors()[0].get_data_tensor()

            if self._gen_product is None:
                # create output tensor from generator (in train set up)
                self._gen_product = self._return_training_graph_from_certain_output(
                    self._generator.get_outputs_maki_tensors()[0]
                )

