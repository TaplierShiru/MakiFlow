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


class BasicTrainingModule(L1orL2LossModuleGenerator,
                          PerceptualLossModuleGenerator):
    """
    Connect additional losses

    """

    def __init__(self, generator, discriminator, name='GeneratorDiscriminator'):
        self._l1_or_l2_loss_vars_are_ready = False
        self._perceptual_loss_vars_are_ready = False
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         name=name
        )

    def _build_additional_losses(self, total_loss):
        if super().is_use_l1_or_l2_loss():
            total_loss += self._build_l1_or_l2_loss()

        if super().is_use_perceptual_loss():
            total_loss += self._build_perceptual_loss()

        return total_loss

