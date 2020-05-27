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


class PerceptualBinaryCETrainingModuleGenerator(GANsBasic):

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
        self._labels = tf.placeholder(dtype=np.float32, shape=[num_labels, 1])
        self._training_vars_are_ready = True


    def _return_training_graph_by_name(self, output:MakiTensor, name_input, input_shape, train=True):
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
                else:
                    input_tensor = InputLayer(input_shape=input_shape, 
                                                    name=f'Input_for_{name_input}').get_data_tensor()
                    output_tensors[f'Input_for_{name_input}'] = input_tensor
                    X = input_tensor

                output_tensors[layer.get_name()] = X
                return X
            else:
                return output_tensors[from_.get_name()]

        training_outputs = create_tensor(output)
        return training_outputs, output_tensors[f'Input_for_{name_input}']

    def add_perceptual_creation_loss(self, create_loss, scale_loss=1e-2, use_perceptual_loss_only=False):
        """
        Add the function that create percetual loss inplace.
        Parameters
        ----------
        creation_per_loss : function
            Function which will create percetual loss. 
            This function must have 3 main input: generated_image, target_image, sess.
            Example of function:
                def create_loss(generated_image, target_image, sess):
                    ...
                    ...
                    return percetual_loss
            Where percetual_loss - is tensorflow Tensor
        """
        self._setup_for_training()
        self._prepare_training_vars()
        self._scale_perceptual_loss = tf.constant(scale_loss, dtype=tf.float32, name='Loss_scale_perceptual')

        self._perceptual_output_fake, self._perceptual_input_fake = self._return_training_graph_by_name(name_input='perceptual_fake_input', 
                                                                output=self._generator._outputs[0],
                                                                input_shape = self._generator._inputs[0].get_shape()
        )
        self._perceptual_input_real = tf.placeholder(dtype=np.float32, shape=self._discriminator._inputs[0].get_shape())
        self._perceptual_loss = True
        self._create_perceptual_loss = create_loss
        self._use_perceptual_loss_only = use_perceptual_loss_only

    def _build_binary_ce_loss(self):

        #self._binary_ce_loss = self._labels * tf.log(1 - self._logits + EPSILON)
        if self._use_perceptual_loss_only:
            self._binary_ce_loss = self._create_perceptual_loss(self._perceptual_output_fake, 
                                                                self._perceptual_input_real, 
                                                                self._session
            ) * self._scale_perceptual_loss
        else:
            binary_ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._labels, logits=self._logits, name='generator_loss')
            binary_ce_loss = tf.reduce_mean(binary_ce_loss)
            perceptual_loss = self._create_perceptual_loss(self._perceptual_output_fake, 
                                                            self._perceptual_input_real, 
                                                            self._session
            )
            self._binary_ce_loss = perceptual_loss * self._scale_perceptual_loss + binary_ce_loss
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

    def fit_perceptual_ce(
            self, Xgen, Xreal, Ytrain, optimizer=None, epochs=1, global_step=None
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
        assert self._perceptual_loss == True, "Perceptual loss is not added!"
        train_op = self._minimize_ce_loss(optimizer, global_step)
        train_costs = []

        try:
            for i in range(epochs):
                if Xgen is not None:
                    generated = Xgen
                else:
                    generated = self._generator.get_noise()

                if self._use_perceptual_loss_only:
                    train_cost_batch, _ = self._session.run(
                        [self._final_binary_ce_loss, train_op],
                        feed_dict={self._perceptual_input_real:Xreal, self._perceptual_input_fake: generated})
                else:
                    train_cost_batch, _ = self._session.run(
                        [self._final_binary_ce_loss, train_op],
                        feed_dict={self._labels: Ytrain, self._images: generated,
                                    self._perceptual_input_real:Xreal, self._perceptual_input_fake: generated})

                train_costs.append(train_cost_batch)

        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            return {PerceptualBinaryCETrainingModuleGenerator.TRAIN_COSTS: train_costs}
