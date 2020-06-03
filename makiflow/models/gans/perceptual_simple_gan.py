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

# TODO: Delete this file and convinced of it

import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np

from makiflow.models.common.utils import moving_average

from .main_modules import GeneratorDiscriminatorBasic

from .training_modules import PerceptualBinaryCETrainingModuleGenerator, BinaryCETrainingModuleDiscriminator

class Discriminator(BinaryCETrainingModuleDiscriminator):
    pass

class GeneratorDiscriminator(PerceptualBinaryCETrainingModuleGenerator, GeneratorDiscriminatorBasic):
    pass

class PerceptualSimpleGAN:

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

    def fit_perceptual_ce(
        self, Xtrain, restore_image_function,
        Xgen=None,
        final_image_size=None, 
        optimizer_discriminator=None, 
        optimizer_generator=None, test_period=1,
        epochs=1, test_period_disc=1,
        global_step=None, show_images=False,
        label_smoothing=0.9,
        epochs_discriminator=1, epochs_generator=1,
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
        n_batches = len(Xtrain) // batch_size

        discriminator_accuracy = []

        iterator = None
        try:
            for i in range(epochs):
                print(f'Epochs is {i}')
                Xtrain = shuffle(Xtrain)

                generator_cost = 0.0
                discriminator_cost = 0.0

                iterator = tqdm(range(n_batches))

                for j in iterator:
                    image_batch = Xtrain[j * batch_size : batch_size * (j + 1)]
                    if Xgen is not None:
                        x_gen_batch = Xgen[j * batch_size : batch_size * (j + 1)]
                    else:
                        x_gen_batch = None

                    generated_images = self._generator.generate(x=x_gen_batch)

                    X_discriminator = np.concatenate([image_batch, generated_images]).astype(np.float32)

                    y_discriminator = np.zeros((2 * batch_size, 1)).astype(np.float32)
                    y_discriminator[:batch_size, :] = label_smoothing
                    #print('Train discriminator...')
                    info_discriminator = self._discriminator.fit_ce(Xtrain=X_discriminator, Ytrain=y_discriminator,
                                                                    optimizer=optimizer_discriminator, epochs=epochs_discriminator,
                                                                    test_period=test_period_disc
                    )
                    if test_period_disc != -1:
                        discriminator_accuracy += info_discriminator[BinaryCETrainingModuleDiscriminator.TRAIN_ACCURACY]
                    disc_cost = info_discriminator[BinaryCETrainingModuleDiscriminator.TRAIN_COSTS]
                    discriminator_cost = moving_average(discriminator_cost, sum(disc_cost) / len(disc_cost), j)

                    #print('Train generator...')
                    y_gen = np.ones((batch_size, 1)).astype(np.float32)
                    gen_cost = self._generator_discriminator.fit_perceptual_ce(x_gen_batch, image_batch, y_gen, epochs=epochs_generator,
                                                                    optimizer=optimizer_generator)[PerceptualBinaryCETrainingModuleGenerator.TRAIN_COSTS]
                    generator_cost = moving_average(generator_cost, sum(gen_cost) / len(gen_cost), j)

                #train_info = [(TRAIN_LOSS, train_cost)]
                # Validating the network on test data
                #print('Before test....')
                if test_period != -1 and i % test_period == 0:
                    print('Test....')
                    if test_period_disc != -1:
                        avg_accuracy = sum(discriminator_accuracy) / len(discriminator_accuracy)
                        print('Average accuracy of discriminator: ', round(avg_accuracy, 5))
                        discriminator_accuracy = []
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
