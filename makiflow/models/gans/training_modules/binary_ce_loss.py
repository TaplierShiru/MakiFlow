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


from ..main_modules import GANsBasic
from .additional_losses import BasicTrainingModule

import tensorflow as tf
from makiflow.models.common.utils import moving_average
from makiflow.models.common.utils import new_optimizer_used, loss_is_built
import numpy as np
from sklearn.utils import shuffle
from makiflow.models.classificator.utils import error_rate
EPSILON = np.float32(1e-32)


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

        super()._prepare_training_vars()

        self._training_vars_are_ready = True


    def _build_binary_ce_loss(self):
        #self._binary_ce_loss = self._labels * tf.log(1 - self._logits + EPSILON)
        self._binary_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._labels,
                                                                       logits=self._logits,
                                                                       name='generator_loss'
        )

        self._binary_ce_loss = tf.reduce_mean(self._binary_ce_loss)
        self._binary_ce_loss = self._build_additional_losses(self._binary_ce_loss)
        self._final_binary_ce_loss = self._build_final_loss(self._binary_ce_loss)

        self._binary_ce_loss_is_built = True

    def _minimize_ce_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._binary_ce_loss_is_built:
            # no need to setup any inputs for this loss
            self._build_binary_ce_loss()
            self._binary_ce_optimizer = optimizer
            self._binary_ce_train_op = optimizer.minimize(
                self._final_binary_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            self._binary_ce_loss_is_built = True
            loss_is_built()

        if self._binary_ce_optimizer != optimizer:
            new_optimizer_used()
            self._binary_ce_optimizer = optimizer
            self._binary_ce_train_op = optimizer.minimize(
                self._final_binary_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._binary_ce_train_op

    def fit_ce(
            self, Xreal=None, Xgen=None, optimizer=None, epochs=1, global_step=None
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
        train_op = self._minimize_ce_loss(optimizer, global_step)

        train_costs = []

        try:
            for i in range(epochs):
                if Xgen is not None:
                    generated = Xgen
                else:
                    generated = self._generator.get_noise()

                if not(self.is_use_l1() or self.is_use_perceptual_loss()):
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


class BinaryCETrainingModuleDiscriminator(GANsBasic):

    TRAIN_COSTS = 'train costs'
    TRAIN_ACCURACY = 'train accuracy'

    def evaluate(self, Xtest, Ytest):
        Xtest = Xtest.astype(np.float32)
        Yish_test = tf.nn.sigmoid(self._inference_out)
        n_batches = Xtest.shape[0] // self._batch_sz

        predictions = np.zeros(len(Xtest))
        for k in range(n_batches):
            Xtestbatch = Xtest[k * self._batch_sz:(k + 1) * self._batch_sz]
            Yish_test_done = self._session.run(Yish_test, feed_dict={self._images: Xtestbatch}) + EPSILON
            predictions[k * self._batch_sz:(k + 1) * self._batch_sz] = Yish_test_done[:, 0]

        Ytest = np.round(Ytest)[:, 0]
        Yish_test_done = np.round(np.array(predictions))
        accuracy_r = 1 - error_rate(Yish_test_done, Ytest)
        return accuracy_r

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()
        # VARIABLES PREPARATION
        self._logits = self._training_outputs[0]
        self._num_classes = self._logits.get_shape()[-1]

        num_labels = self._logits.get_shape()[0]
        self._images = self._input_data_tensors[0]
        self._labels = tf.placeholder(dtype=np.float32, shape=self._logits.get_shape().as_list())
        self._batch_sz = num_labels
        self._training_vars_are_ready = True

    def _build_binary_ce_loss(self):
        #self._binary_ce_loss = -(self._labels * tf.log(self._logits) + (1 - self._labels) * tf.log(1 - self._logits))
        self._binary_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._labels, logits=self._logits, name='sigmoid_loss')
        binary_ce_loss = tf.reduce_mean(self._binary_ce_loss)
        self._final_binary_ce_loss = self._build_final_loss(binary_ce_loss)

        self._binary_ce_loss_is_built = True

    def _minimize_ce_loss(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._binary_ce_loss_is_built:
            # no need to setup any inputs for this loss
            self._build_binary_ce_loss()
            self._binary_ce_optimizer = optimizer
            self._binary_ce_train_op = optimizer.minimize(
                self._final_binary_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            self._binary_ce_loss_is_built = True
            loss_is_built()

        if self._binary_ce_optimizer != optimizer:
            new_optimizer_used()
            self._binary_ce_optimizer = optimizer
            self._binary_ce_train_op = optimizer.minimize(
                self._final_binary_ce_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._binary_ce_train_op

    def fit_ce(
            self, Xtrain, Ytrain, optimizer=None, epochs=1, test_period=1, global_step=None
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
        train_op = self._minimize_ce_loss(optimizer, global_step)

        n_batches = len(Xtrain) // self._batch_sz

        train_costs = []
        train_accuracy = []

        try:
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                train_cost = 0

                for j in range(n_batches):
                    Xbatch = Xtrain[j * self._batch_sz:(j + 1) * self._batch_sz]
                    Ybatch = Ytrain[j * self._batch_sz:(j + 1) * self._batch_sz]
                    train_cost_batch, _ = self._session.run(
                        [self._final_binary_ce_loss, train_op],
                        feed_dict={self._images: Xbatch, self._labels: Ybatch})
                    # Use exponential decay for calculating loss and error
                    train_cost = moving_average(train_cost, train_cost_batch, j)

                train_costs.append(train_cost)
                # Validating the network on test data
                if test_period != -1 and i % test_period == 0:
                    # For test data
                    train_accuracy_single = self.evaluate(Xtrain, Ytrain)
                    train_accuracy.append(train_accuracy_single)

        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            return {BinaryCETrainingModuleDiscriminator.TRAIN_COSTS: train_costs, 
                    BinaryCETrainingModuleDiscriminator.TRAIN_ACCURACY: train_accuracy}


