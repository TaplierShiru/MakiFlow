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
from makiflow.models.gans.core import DiscriminatorTrainer
from makiflow.core import TrainerBuilder


class BinaryCETrainerDisc(DiscriminatorTrainer):
    TYPE = 'BinaryCETrainerGen'

    BINARY_CE_DISC_LOSS = 'BINARY_CE_DISC_LOSS'

    def _build_local_loss(self, prediction, label):
        # self._binary_ce_loss = -(self._labels * tf.log(self._logits) + (1 - self._labels) * tf.log(1 - self._logits))
        final_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label, logits=prediction,
            name=BinaryCETrainerDisc.BINARY_CE_DISC_LOSS
        )
        final_loss = tf.reduce_mean(final_loss)
        return final_loss


TrainerBuilder.register_trainer(BinaryCETrainerDisc)