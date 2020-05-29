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

from .additional_losses import BasicTrainingModule

import tensorflow as tf
from makiflow.models.common.utils import new_optimizer_used, loss_is_built
import numpy as np
from makiflow.base import MakiTensor


class FeatureBinaryCETrainingModuleGenerator(BasicTrainingModule):
    TRAIN_COSTS = 'train costs'
    GEN_LABEL = 'gen_label'

    def _prepare_training_vars(self):
        if not self._set_for_training:
            super()._setup_for_training()
        # VARIABLES PREPARATION
        self._feature_matching_loss_is_built = False

        self._logits = self._training_outputs[0]
        self._num_classes = self._logits.get_shape()[-1]

        num_labels = self._logits.get_shape()[0]
        self._images = self._input_data_tensors[0]
        self._batch_sz = num_labels
        self._labels = tf.constant(np.ones(self._logits.get_shape().as_list()),
                                   shape=self._logits.get_shape().as_list(),
                                   dtype=tf.float32,
                                   name=FeatureBinaryCETrainingModuleGenerator.GEN_LABEL
        )

        self._feature_output_fake = super()._return_training_graph_from_certain_output(
            name_layer_return=self._layer_tensor_feature_disc.get_name(),
            output=self._outputs[0],
        )
        self._feature_output_real, self._input_real_image = (self._layer_tensor_feature_disc.get_data_tensor(),
                                                             self._discriminator.get_inputs_maki_tensors()[0].get_data_tensor())

        super()._prepare_training_vars()

        self._training_vars_are_ready = True

    def add_feature_matching(self, layer_tensor_feature_disc: MakiTensor,
                             use_feature_loss_only=False,
                             feature_scale=1.0,
                             ce_scale=1.0):
        """
        Added layers for feature matching in the loss creation.
        In this model, this function is important before training.

        Parameters
        ----------
        layer_tensor_feature_disc : MakiTensor
            MakiTensor from which will be taken output for feature matching loss.
        use_feature_loss_only : bool
            If true, only feature loss will be used, otherwise also will be added binary ce loss.
        feature_scale : float
            Scale of the feature loss, by default equal to 1.0.
        ce_scale : float
            Scale of the ce loss, by default equal to 1.0.
        """
        # `layer_name_disc` will be used in the L2 loss for feature matching
        self._layer_tensor_feature_disc = layer_tensor_feature_disc
        self._feature_loss_is_set = True
        self._feature_scale = feature_scale
        self._ce_scale = ce_scale
        self._use_feature_loss_only = use_feature_loss_only

    def _build_feature_matching_loss(self):

        # self._feature_matching_loss = self._labels * tf.log(1 - self._logits + EPSILON)
        if self._use_feature_loss_only:
            self._feature_matching_loss = tf.reduce_mean(
                tf.square(
                    tf.reduce_mean(self._feature_output_fake, axis=0) - tf.reduce_mean(self._feature_output_real,
                                                                                       axis=0)
                )
            ) * self._feature_scale
        else:
            # TODO: Delete these self variable, keep them like local. In this case its just for debugging stuff.
            self._ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._labels,
                                                                    logits=self._logits,
                                                                    name='generator_loss'
            )

            self._feature_loss = tf.reduce_mean(
                tf.square(
                    tf.reduce_mean(self._feature_output_fake, axis=0) - \
                    tf.reduce_mean(self._feature_output_real, axis=0)
                )
            )

            self._feature_matching_loss = tf.reduce_mean(self._ce_loss) * self._ce_scale + \
                                          self._feature_loss * self._feature_scale

        self._feature_matching_loss = self._build_additional_losses(self._feature_matching_loss)

        self._final_feature_matching_loss = self._build_final_loss(self._feature_matching_loss)

        self._feature_matching_loss_is_built = True

    def _minimize_feature_matching_ce_loss(self, optimizer, global_step):
        if not self._set_for_training:
            self._setup_for_training()

        if not self._training_vars_are_ready:
            self._prepare_training_vars()

        if not self._feature_matching_loss_is_built:
            # no need to setup any inputs for this loss
            self._build_feature_matching_loss()
            self._feature_optimizer = optimizer
            self._feature_train_op = optimizer.minimize(
                self._final_feature_matching_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))
            loss_is_built()

        if self._feature_optimizer != optimizer:
            new_optimizer_used()
            self._feature_optimizer = optimizer
            self._feature_train_op = optimizer.minimize(
                self._final_feature_matching_loss, var_list=self._trainable_vars, global_step=global_step
            )
            self._session.run(tf.variables_initializer(optimizer.variables()))

        return self._feature_train_op

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

        assert optimizer is not None
        assert self._session is not None
        assert self._feature_loss_is_set, "Feature loss is not added!"
        train_op = self._minimize_feature_matching_ce_loss(optimizer, global_step)
        train_costs = []

        try:
            for i in range(epochs):
                if Xgen is not None:
                    generated = Xgen
                else:
                    generated = self._generator.get_noise()

                train_cost_batch, _ = self._session.run(
                    [self._final_feature_matching_loss, train_op],
                    feed_dict={self._images: generated, self._input_real_image: Xreal}
                )

                train_costs.append(train_cost_batch)

        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            return {FeatureBinaryCETrainingModuleGenerator.TRAIN_COSTS: train_costs}
