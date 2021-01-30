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

import numpy as np
import tensorflow as tf
from makiflow.models.gans.core import GeneratorTrainer
from makiflow.core import TrainerBuilder


class BinaryCETrainerGen(GeneratorTrainer):
    TYPE = 'BinaryCETrainerGen'

    BinaryCETrainerGen = 'BinaryCETrainerGen'

    def _build_local_loss(self, prediction, label):
        # In most cases label must be array of 1's

        # labels * tf.log(1 - logits + EPSILON)
        final_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label, logits=prediction,
            name=BinaryCETrainerGen.BinaryCETrainerGen
        )
        final_loss = tf.reduce_mean(final_loss)
        return final_loss


TrainerBuilder.register_trainer(BinaryCETrainerGen)

"""
class BinaryCETrainingModuleGenerator(BasicTrainingModule):

    TRAIN_COSTS = 'train costs'
    GEN_LABEL = 'gen_label'

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()
        # VARIABLES PREPARATION
        self._binary_ce_loss_is_built = False

        self._logits = self._training_outputs[0]
        self._num_classes = self._logits.get_shape()[-1]

        num_labels = self._logits.get_shape()[0]
        self._images = self._input_data_tensors[0]
        self._batch_sz = num_labels
        self._labels = tf.constant(np.ones(self._logits.get_shape().as_list()),
                                   shape=self._logits.get_shape().as_list(),
                                   dtype=tf.float32,
                                   name=BinaryCETrainingModuleGenerator.GEN_LABEL
        )

        self._training_vars_are_ready = True

    def _build_binary_ce_loss(self):
        #self._binary_ce_loss = self._labels * tf.log(1 - self._logits + EPSILON)
        self._binary_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._labels,
                                                                       logits=self._logits,
                                                                       name='generator_loss'
        )

        self._binary_ce_loss = tf.reduce_mean(self._binary_ce_loss)
        self._binary_ce_loss = super()._build_additional_losses(self._binary_ce_loss)
        self._final_binary_ce_loss = self._build_final_loss(self._binary_ce_loss)

        self._binary_ce_loss_is_built = True

    def fit_ce(
            self, Xreal=None, Xgen=None, optimizer=None, epochs=1, global_step=None
    ):
        assert (optimizer is not None)
        assert (self._session is not None)
        train_op = self._minimize_ce_loss(optimizer, global_step)

        train_costs = []

        try:
            for i in range(epochs):
                # TODO: Add batched iteration and shuffle for data if it's needed
                if Xgen is not None:
                    generated = Xgen
                else:
                    generated = self._generator.get_noise()

                if not(super().is_use_l1_or_l2_loss() or \
                       super().is_use_perceptual_loss()
                ):
                    train_cost_batch, _ = self._session.run(
                        [self._final_binary_ce_loss, train_op],
                        feed_dict={self._images: generated}
                    )
                else:
                    train_cost_batch, _ = self._session.run(
                        [self._final_binary_ce_loss, train_op],
                        feed_dict={self._images: generated,
                                   self._input_real_image: Xreal}
                    )

                train_costs.append(train_cost_batch)

        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            return {BinaryCETrainingModuleGenerator.TRAIN_COSTS: train_costs}

"""