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
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np

from makiflow.models.common.utils import moving_average

from .main_modules import GeneratorDiscriminatorBasic

from .training_modules import WassersteinTrainingModuleGenerator, WassersteinTrainingModuleDiscriminator

from makiflow.layers import ConvLayer, UpConvLayer, DenseLayer


class Discriminator(WassersteinTrainingModuleDiscriminator):

    def __init__(self, input_s, output, generator, name='W_Discriminator', clip=(-0.01, 0.01)):
        self._generator = generator

        super().__init__(input_s=input_s, output=output, name=name)

        # set clip for all weights in conv/upconv/dense layers
        self._clip_op = []
        """
        for name, maki_tensor in self._graph_tensors.items():
            if maki_tensor.get_parent_layer().TYPE == ConvLayer.TYPE or \
                maki_tensor.get_parent_layer().TYPE == UpConvLayer.TYPE or \
                maki_tensor.get_parent_layer().TYPE == DenseLayer.TYPE:

                W = maki_tensor.get_parent_layer().W
                self._clip_op += [W.assign(tf.clip_by_value(W, clip[0], clip[1]))]
        """
        for name, maki_tensor in self._graph_tensors.items():
            params = list(maki_tensor.get_parent_layer().get_params_dict().values())
            for param in params:
                self._clip_op += [param.assign(tf.clip_by_value(param, clip[0], clip[1]))]



class GeneratorDiscriminator(WassersteinTrainingModuleGenerator, GeneratorDiscriminatorBasic):
    pass


class WGAN:

    def __init__(self, generator, discriminator, generator_discriminator):
        self._discriminator = discriminator
        self._generator = generator
        # Set layers from generator in disciminator as untrainable
        untrain = []
        generator_names = list(generator.get_layers_names().keys())
        for name in generator_discriminator.get_layers_names().keys():
            if name not in generator_names:
                untrain.append((name, False))

        generator_discriminator.set_layers_trainable(untrain)

        self._generator_discriminator = generator_discriminator

    def set_session(self, sess : tf.Session):
        self._session = sess

        self._generator_discriminator.set_session(sess)
        self._discriminator._session = sess
        self._generator._session = sess

    def fit_w(
        self, Xtrain, restore_image_function,
        Xgen=None,
        final_image_size=None, 
        optimizer_discriminator=None, 
        optimizer_generator=None, test_period=1,
        epochs=1,
        global_step=None, show_images=False,
        epochs_discriminator=1, epochs_generator=1,
        count_disc_batches=5,
        ):
        """
        Method for training the model. Works faster than `verbose_fit` method because
        it uses exponential decay in order to speed up training. It produces less accurate
        train error measurement.

        Parameters
        ----------
        Xtrain : numpy array
            Training images stacked into one big array with shape (num_images, image_w, image_h, image_depth).
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

        assert (optimizer_discriminator is not None)
        assert (optimizer_generator is not None)
        assert (self._session is not None)

        batch_size = self._generator_discriminator.get_input_shape()[0]

        n_batches = len(Xtrain) // (batch_size * count_disc_batches)

        iterator = None
        try:
            for i in range(epochs):
                print(f'Epochs is {i}')
                shuffle(Xtrain)
                generator_cost = 0.0
                discriminator_cost = 0.0


                iterator = tqdm(range(n_batches))

                for j in iterator:
                    # train discriminator
                    x_batch = Xtrain[j * batch_size * count_disc_batches: (j + 1) * batch_size * count_disc_batches]
                    if Xgen is not None:
                        x_gen_batch = Xgen[j * batch_size* count_disc_batches : batch_size * (j + 1) * count_disc_batches]
                    else:
                        x_gen_batch = None

                    disc_cost = self._discriminator.fit_wasserstein(x_gen_batch, Xtrain=x_batch,
                                                                    optimizer=optimizer_discriminator, epochs=epochs_discriminator
                    )[WassersteinTrainingModuleDiscriminator.TRAIN_COSTS]

                    discriminator_cost = moving_average(discriminator_cost, sum(disc_cost) / len(disc_cost), j)

                    if x_gen_batch is not None:
                        x_gen_batch = np.array(x_gen_batch)[np.random.randint(0, len(x_gen_batch), size=batch_size)]
                    # train generator
                    gen_cost = self._generator_discriminator.fit_wasserstein(epochs=epochs_generator,
                                                                    optimizer=optimizer_generator)[WassersteinTrainingModuleGenerator.TRAIN_COSTS]
                    generator_cost = moving_average(generator_cost, sum(gen_cost) / len(gen_cost), j)

                #train_info = [(TRAIN_LOSS, train_cost)]
                # Validating the network on test data
                #print('Before test....')
                if test_period != -1 and i % test_period == 0:
                    print('Test....')

                    print('Average costs of dicriminator: ', discriminator_cost)
                    print('Average costs of generator: ', generator_cost)

                    generated_images = self._generator.generate()

                    if final_image_size is not None:
                        generated_images = generated_images.reshape(generated_images.shape[0], *final_image_size)
                    
                    generated_images = np.clip(restore_image_function(generated_images), 0.0, 255.0).astype(np.uint8)
                    
                    plt.figure(figsize=(20, 20))
                    for z in range(min(batch_size, 100)):
                        plt.subplot(10, 10, z+1)
                        plt.imshow(generated_images[z])
                        plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(f'generated_image_epoch_{i}.png')
                    if show_images:
                        plt.show()

                    plt.close('all')

        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
