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


from ..main_modules import AdversarialBasic

import tensorflow as tf
from makiflow.models.common.utils import new_optimizer_used, loss_is_built
from makiflow.base.loss_builder import Loss
from makiflow.generators.nn_render import NNRIterator
import numpy as np
from sklearn.utils import shuffle


class AdversarialTrainingModule(AdversarialBasic):

    TRAIN_COSTS = 'train costs'
    GEN_LABEL = 'gen_label'

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()
        # VARIABLES PREPARATION
        self._adversarial_loss_is_built = False

        self._logits = self._training_outputs[0]
        self._num_classes = self._logits.get_shape()[-1]

        num_labels = self._logits.get_shape()[0]
        self._images = self._input_data_tensors[0]
        self._batch_sz = num_labels
        self._labels = tf.constant(
            np.ones(self._logits.get_shape().as_list()),
            shape=self._logits.get_shape().as_list(),
            dtype=tf.float32,
            name=AdversarialTrainingModule.GEN_LABEL
        )

        self._input_real_image = self._discriminator_model.get_inputs_maki_tensors()[0].get_data_tensor()

        # create output tensor from generator (in train set up)
        self._gen_product = self._return_training_graph_from_certain_output(
            self._generator_model.get_outputs_maki_tensors()[0]
        )

        self._training_vars_are_ready = True

    def _setup_masked_adversarial_loss_inputs(self):
        if self._use_mask:
            self._adv_mask = self._discriminator_model.get_iterator()[NNRIterator.BIN_MASK]

    def _build_adversarial_loss(self):
        adversarial_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self._labels,
            logits=self._logits,
            name='generator_loss'
        )
        adversarial_loss = tf.reduce_mean(adversarial_loss)

        if self._use_l1:
            if self._use_mask:
                diff_loss = Loss.abs_loss(self._input_real_image, self._gen_product, raw_tensor=True)
                diff_loss = tf.reduce_sum(diff_loss * self._adv_mask)
                diff_loss = diff_loss / tf.reduce_sum(self._adv_mask)
            else:
                diff_loss = Loss.abs_loss(self._input_real_image, self._gen_product)
        else:
            if self._use_mask:
                diff_loss = Loss.mse_loss(self._input_real_image, self._gen_product, raw_tensor=True)
                diff_loss = tf.reduce_sum(diff_loss * self._adv_mask)
                diff_loss = diff_loss / tf.reduce_sum(self._adv_mask)
            else:
                diff_loss = Loss.mse_loss(self._input_real_image, self._gen_product)

        final_adversarial_loss = adversarial_loss + diff_loss * self._lambda
        self._final_adversarial_loss = self._build_final_loss(final_adversarial_loss)

        self._adversarial_loss_is_built = True

    def _minimize_adversarial_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._adversarial_loss_is_built:
            # no need to setup any inputs for this loss
            self._setup_masked_adversarial_loss_inputs()
            self._build_adversarial_loss()
            self._adversarial_optimizer = optimizer
            self._adversarial_train_op = optimizer.minimize(
                self._final_adversarial_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            loss_is_built()

        if self._adversarial_optimizer != optimizer:
            new_optimizer_used()
            self._adversarial_optimizer = optimizer
            self._adversarial_train_op = optimizer.minimize(
                self._final_adversarial_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._adversarial_train_op

    def fit_ce(
            self, Xreal, Xgen, optimizer=None, epochs=1, global_step=None
    ):
        """
        Method for training the model. Works faster than `verbose_fit` method because
        it uses exponential decay in order to speed up training. It produces less accurate
        train error measurement.

        Parameters
        ----------
        Xtrain : numpy array
            Training images stacked into one big array with shape (num_images, image_w, image_h, image_depth).
        Ytrain : numpy array
            Training label for each image in `Xtrain` array with shape (num_images).
            IMPORTANT: ALL LABELS MUST BE NOT ONE-HOT ENCODED, USE SPARSE TRAINING DATA INSTEAD.
        Xtest : numpy array
            Same as `Xtrain` but for testing.
        Ytest : numpy array
            Same as `Ytrain` but for testing.
        optimizer : tensorflow optimizer
            Model uses tensorflow optimizers in order train itself.
        epochs : int
            Number of epochs.
        test_period : int
            Test begins each `test_period` epochs. You can set a larger number in order to
            speed up training.

        Returns
        -------
            python dictionary
                Dictionary with all testing data(train error, train cost, test error, test cost)
                for each test period.
        """

        assert (optimizer is not None)
        assert (self._session is not None)
        train_op = self._minimize_adversarial_loss(optimizer, global_step)

        train_costs = []
        n_batches = len(Xreal) // self._batch_sz

        try:
            for i in range(epochs):
                for j in range(n_batches):
                    generated_batch = Xgen[j * self._batch_sz:(j + 1) * self._batch_sz]
                    Xreal_batch = Xreal[j * self._batch_sz:(j + 1) * self._batch_sz]

                    train_cost_batch, _ = self._session.run(
                        [self._final_adversarial_loss, train_op],
                        feed_dict={self._images: generated_batch,
                                   self._input_real_image: Xreal_batch}
                    )

                    train_costs.append(train_cost_batch)

        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            return {AdversarialTrainingModule.TRAIN_COSTS: train_costs}

