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
from makiflow.base import MakiTensor

from copy import copy

EPSILON = np.float32(1e-32)


class FeatureBinaryCETrainingModuleGenerator(GANsBasic):

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
        self._labels = tf.constant(np.ones(self._logits.get_shape().as_list()),
                                   shape=self._logits.get_shape().as_list(),
                                   dtype=tf.float32,
                                   name='gen_label'
        )

        # prepare inputs and outputs for l1 or l2 if it need
        if self._use_l1 is not None:
            # create output tensor from generator (in train set up)
            self._gen_product = self._return_training_graph_from_certain_output(self._generator._outputs[0])

        self._training_vars_are_ready = True

    def _return_training_graph_by_name_at_certain_tensor(self, name_layer,
                                                         output:MakiTensor,
                                                         name_input, input_shape,
                                                         train=False):
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
                if len(from_.get_parent_tensor_names()) != 0:
                    for elem in from_.get_parent_tensors():
                        takes += [create_tensor(elem)]

                    if layer.get_name() in self._trainable_layers and train:
                        X = layer._training_forward(takes[0] if len(takes) == 1 else takes)
                    else:
                        X = layer._forward(takes[0] if len(takes) == 1 else takes)

                output_tensors[layer.get_name()] = X
                return X
            else:
                return output_tensors[from_.get_name()]

        _ = create_tensor(output)
        return output_tensors[name_layer]

    def add_feature_matching(self, layer_tensor_feature_disc: MakiTensor, use_feature_loss_only=False):
        # `layer_name_disc` will be used in the L2 loss for feature matching
        self._setup_for_training()
        self._prepare_training_vars()

        self._feature_output_fake = self._return_training_graph_by_name_at_certain_tensor(layer_tensor_feature_disc.get_name(),
                                                                name_input='feature_fake_input', 
                                                                output=self._outputs[0],
                                                                input_shape = self._inputs[0].get_shape()
        )
        self._feature_output_real, self._feature_input_real = (layer_tensor_feature_disc.get_data_tensor(),
                                                              self._discriminator._inputs[0].get_data_tensor())
        
        self._feature_loss = True
        self._use_feature_loss_only = use_feature_loss_only

    def _build_binary_ce_loss(self):

        #self._binary_ce_loss = self._labels * tf.log(1 - self._logits + EPSILON)
        if self._use_feature_loss_only:
            self._binary_ce_loss = tf.reduce_mean(
                           tf.square(
                           tf.reduce_mean(self._feature_output_fake, axis = 0) - tf.reduce_mean(self._feature_output_real, axis = 0))
            )
        else:
            self._binary_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._labels,
                                                                           logits=self._logits, name='generator_loss')
            self._binary_ce_loss = tf.reduce_mean(self._binary_ce_loss)
            if self._feature_loss:
                feature_loss = tf.reduce_mean(
                               tf.square(
                               tf.reduce_mean(self._feature_output_fake, axis = 0) - tf.reduce_mean(self._feature_output_real, axis = 0))
                )
                self._binary_ce_loss += feature_loss

        # additional loss l1/l2
        if self._use_l1 is not None:
            if self._use_l1:
                # build l1
                additional_loss = tf.reduce_mean(tf.abs(self._gen_product - self._feature_input_real)) * self._lambda
            else:
                # build l2
                additional_loss = tf.reduce_mean(
                                  tf.square(self._gen_product - self._feature_input_real)
                ) * 0.5 * self._lambda
            # add aditional loss to final loss
            self._binary_ce_loss += additional_loss

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

    def fit_feature_ce(
            self, Xgen, Xreal, optimizer=None, epochs=1, global_step=None
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
        assert self._feature_loss == True, "Feature loss is not added!"
        train_op = self._minimize_ce_loss(optimizer, global_step)
        train_costs = []

        try:
            for i in range(epochs):
                if Xgen is not None:
                    generated = Xgen
                else:
                    generated = self._generator.get_noise()

                train_cost_batch, _ = self._session.run(
                    [self._final_binary_ce_loss, train_op],
                    feed_dict={self._images: generated, self._feature_input_real: Xreal}
                )

                train_costs.append(train_cost_batch)

        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            return {FeatureBinaryCETrainingModuleGenerator.TRAIN_COSTS: train_costs}
