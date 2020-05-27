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

import tensorflow as tf
from makiflow.models.common.utils import print_train_info, moving_average
from makiflow.models.common.utils import new_optimizer_used, loss_is_built
import numpy as np
from sklearn.utils import shuffle
from makiflow.models.classificator.utils import error_rate, sparse_cross_entropy
from makiflow.layers import InputLayer
from copy import copy
EPSILON = np.float32(1e-32)


class WassersteinTrainingModuleGenerator(GANsBasic):

    TRAIN_COSTS = 'train costs'

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()
        # VARIABLES PREPARATION
        self._logits = self._training_outputs[0]
        self._num_classes = self._logits.get_shape()[-1]

        num_labels = self._logits.get_shape()[0]
        self._images = self._input_data_tensors[0]
        self._batch_sz = num_labels
        self._training_vars_are_ready = True

    def _build_wasserstein_loss(self):

        self._wasserstein_loss = (-1) * tf.reduce_mean(self._logits)
        self._final_wasserstein_loss = self._build_final_loss(self._wasserstein_loss)

        self._wasserstein_is_built = True

    def _minimize_wasserstein(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._wasserstein_is_built:
            # no need to setup any inputs for this loss
            self._build_wasserstein_loss()
            self._wasserstein_optimizer = optimizer
            self._wasserstein_train_op = optimizer.minimize(
                self._final_wasserstein_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            self._wasserstein_is_built = True
            loss_is_built()

        if self._wasserstein_optimizer != optimizer:
            new_optimizer_used()
            self._wasserstein_optimizer = optimizer
            self._wasserstein_train_op = optimizer.minimize(
                self._final_wasserstein_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._wasserstein_train_op

    def fit_wasserstein(
            self, Xgan, optimizer=None, epochs=1, global_step=None
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
        train_op = self._minimize_wasserstein(optimizer, global_step)

        train_costs = []

        try:
            for i in range(epochs):
                if Xgen is not None:
                    generated = Xgen
                else:
                    generated = self._generator.get_noise()

                train_cost_batch, _ = self._session.run(
                    [self._final_wasserstein_loss, train_op],
                    feed_dict={self._images: generated})

                train_costs.append(train_cost_batch)

        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            return {WassersteinTrainingModuleGenerator.TRAIN_COSTS: train_costs}


class WassersteinTrainingModuleDiscriminator(GANsBasic):

    TRAIN_COSTS = 'train costs'

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

    def _return_training_graph_fake_logits(self):
        # Contains pairs {layer_name: tensor}, where `tensor` is output
        # tensor of layer called `layer_name`
        output_tensors = {}
        used = {}

        def create_tensor(from_):
            if used.get(from_.get_name()) is None:
                layer = from_.get_parent_layer()
                used[layer.get_name()] = True
                X = copy(from_.get_data_tensor())
                takes = []
                # Check if we at the beginning of the computational graph, i.e. InputLayer
                if from_.get_parent_tensor_names() is not None:
                    for elem in from_.get_parent_tensors():
                        takes += [create_tensor(elem)]

                    if layer.get_name() in self._trainable_layers:
                        X = layer._training_forward(takes[0] if len(takes) == 1 else takes)
                    else:
                        X = layer._forward(takes[0] if len(takes) == 1 else takes)
                else:
                    self._fake_input = InputLayer(input_shape=self._input_data_tensors[0].get_shape().as_list(), 
                                                    name='Input_for_fake_logits').get_data_tensor()
                    X = self._fake_input

                output_tensors[layer.get_name()] = X
                return X
            else:
                return output_tensors[from_.get_name()]

        training_outputs = []
        for output in self._outputs:
            training_outputs += [create_tensor(output)]

        return training_outputs

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()
        # VARIABLES PREPARATION
        self._real_logits = self._training_outputs[0]

        self._real_input = self._input_data_tensors[0]
        # For fake logits we have self._fake_input input
        self._fake_logits = self._return_training_graph_fake_logits()[0]
        self._batch_sz = self._real_logits.get_shape()[0]
        self._training_vars_are_ready = True

    def _build_wasserstein_loss(self):
        self._wasserstein_loss = tf.reduce_mean(self._real_logits) - tf.reduce_mean(self._fake_logits)
        self._final_wasserstein_loss = self._build_final_loss(self._wasserstein_loss)

        self._wasserstein_is_built = True

    def _minimize_wasserstein(self, optimizer, global_step):
        if not self._set_for_training:
            super()._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._wasserstein_is_built:
            # no need to setup any inputs for this loss
            self._build_wasserstein_loss()
            self._wasserstein_optimizer = optimizer
            self._wasserstein_train_op = optimizer.minimize(
                (-1) * self._final_wasserstein_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            self._wasserstein_is_built = True
            loss_is_built()

        if self._wasserstein_optimizer != optimizer:
            new_optimizer_used()
            self._wasserstein_optimizer = optimizer
            self._wasserstein_train_op = optimizer.minimize(
                (-1) * self._final_wasserstein_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._wasserstein_train_op

    def fit_wasserstein(
            self, Xgan, Xtrain, optimizer=None, epochs=1, global_step=None,
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
        train_op = self._minimize_wasserstein(optimizer, global_step)


        n_batches = len(Xtrain) // self._batch_sz

        train_costs = []
        train_accuracy = []

        try:
            for i in range(epochs):
                train_cost = 0

                for j in range(n_batches):
                    real_batch = Xtrain[j * self._batch_sz:(j + 1) * self._batch_sz]
                    if Xgen is not None:
                        fake_batch = self._generator.generate(Xgen[j * self._batch_sz:(j + 1) * self._batch_sz])
                    else:
                        fake_batch = self._generator.generate(x=Xgen)

                    train_cost_batch, _ = self._session.run(
                        [self._final_wasserstein_loss, train_op],
                        feed_dict={self._fake_input: fake_batch, self._real_input: real_batch})

                    # clip all weight in (-c, c) range
                    self._session.run(self._clip_op)
                    # Use exponential decay for calculating loss and error
                    train_cost = moving_average(train_cost, train_cost_batch, j)

                train_costs.append(train_cost)

        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            return {WassersteinTrainingModuleDiscriminator.TRAIN_COSTS: train_costs}


