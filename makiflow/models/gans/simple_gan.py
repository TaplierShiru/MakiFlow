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
import cv2

from makiflow.models.common.utils import moving_average

from .main_modules import GeneratorDiscriminatorBasic, GANsBasic

from .training_modules import BinaryCETrainingModuleGenerator, BinaryCETrainingModuleDiscriminator

from .pipeline.input_gen_layer import InputGenLayer


class Discriminator(BinaryCETrainingModuleDiscriminator):
    pass


class GeneratorDiscriminator(BinaryCETrainingModuleGenerator, GeneratorDiscriminatorBasic):
    pass


class SimpleGAN:

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
        self._session = None
        self._generator_discriminator = generator_discriminator
        # generator(pipeline) variable
        self._generator_is_set = False
        self._gen_in_target = None
        self._gen_in_input = None

    def set_session(self, sess : tf.Session):
        self._session = sess

        self._generator_discriminator.set_session(sess)
        self._discriminator._session = sess
        self._generator._session = sess

    def set_generator(self, gen_in_target: InputGenLayer, gen_in_input: InputGenLayer):
        """
        Set generators for training.

        Parameters
        ----------
        gen_in_target : InputGenLayer
            Is target images for generator.
        gen_in_input : InputGenLayer
            Is input images for generator. If your model use noise as input, just set to None.
        """
        self._generator_is_set = True
        self._gen_in_target = gen_in_target
        self._gen_in_input = gen_in_input

    def genfit_ce(
        self, iterations, restore_image_function,
        final_image_size=None, 
        optimizer_discriminator=None, 
        optimizer_generator=None, test_period=1,
        epochs=1, test_period_disc=1,
        global_step_disc=None, global_step_gen=None,
        show_images=False,
        label_smoothing=0.9,
        use_BGR2RGB=False,
        pre_donwload=100,
        epochs_discriminator=1, epochs_generator=1,
        ):
        """

        Parameters
        ----------

        Returns
        -------
            python dictionary
                Dictionary with all testing data(train error, train cost, test error, test cost)
                for each test period.
        """

        assert (optimizer_discriminator is not None)
        assert (optimizer_generator is not None)
        assert (self._session is not None)
        assert (self._generator_is_set)

        batch_size = self._generator_discriminator.get_input_shape()[0]

        discriminator_accuracy = []
        iterations = int(iterations / pre_donwload) * pre_donwload
        y_discriminator = np.zeros((2 * batch_size, *self._discriminator.get_logits_shape()[1:])).astype(np.float32)
        y_discriminator[:batch_size, ...] = label_smoothing

        iterator = None
        try:
            for i in range(epochs):
                print(f'Epochs is {i}')

                generator_cost = 0.0
                discriminator_cost = 0.0
                iterator = tqdm(range(iterations))

                image_batch_preload = []
                x_gen_batch_preload = []
                x_gen_batch = None
                current_batch_preload = 0

                for j in iterator:
                    # specific input for generator
                    """
                    if self._gen_in_input is not None:
                        # if we separate this, generator will provide different images
                        x_gen_batch, image_batch = self._session.run([self._gen_in_input.get_data_tensor(),
                                                                      self._gen_in_target.get_data_tensor()]
                        )
                    else:
                        image_batch = self._session.run(self._gen_in_target.get_data_tensor())
                    """
                    if current_batch_preload == pre_donwload or len(image_batch_preload) == 0:
                        current_batch_preload = 0
                        x_gen_batch_preload = []
                        image_batch_preload = []

                        for _ in range(pre_donwload):
                            if self._gen_in_input is not None:
                                # if we separate this, generator will provide different images
                                x_gen_batch_single, image_batch_single = self._session.run([self._gen_in_input.get_data_tensor(),
                                                                              self._gen_in_target.get_data_tensor()]
                                )
                                x_gen_batch_preload.append(x_gen_batch_single)
                            else:
                                image_batch_single = self._session.run(self._gen_in_target.get_data_tensor())
                            image_batch_preload.append(image_batch_single)

                    image_batch = image_batch_preload[current_batch_preload]
                    # specific input for generator
                    if self._gen_in_input is not None:
                        x_gen_batch = x_gen_batch_preload[current_batch_preload]
                    else:
                        x_gen_batch = None

                    current_batch_preload += 1

                    generated_images = self._generator.generate(x=x_gen_batch)

                    X_discriminator = np.concatenate([image_batch, generated_images]).astype(np.float32)
                    # Train discriminator
                    info_discriminator = self._discriminator.fit_ce(Xtrain=X_discriminator,
                                                                    Ytrain=y_discriminator,
                                                                    optimizer=optimizer_discriminator,
                                                                    epochs=epochs_discriminator,
                                                                    test_period=test_period_disc,
                                                                    global_step=global_step_disc
                    )
                    if test_period_disc != -1:
                        discriminator_accuracy += info_discriminator[BinaryCETrainingModuleDiscriminator.TRAIN_ACCURACY]
                    disc_cost = info_discriminator[BinaryCETrainingModuleDiscriminator.TRAIN_COSTS]
                    discriminator_cost = moving_average(discriminator_cost, sum(disc_cost) / len(disc_cost), j)

                    # Train generator
                    # if user use l1 or l2 loss
                    if self._generator_discriminator._use_l1 is not None:
                        Xreal = image_batch
                    else:
                        Xreal = None
                        
                    gen_cost = self._generator_discriminator.fit_ce(Xreal=Xreal, Xgen=x_gen_batch,
                                                                    epochs=epochs_generator,
                                                                    optimizer=optimizer_generator,
                                                                    global_step=global_step_gen
                    )[BinaryCETrainingModuleGenerator.TRAIN_COSTS]
                    generator_cost = moving_average(generator_cost, sum(gen_cost) / len(gen_cost), j)

                # close tqdm iterator for out safe
                iterator.close()
                # Validating the network on test data
                if test_period != -1 and i % test_period == 0:
                    print('Test....')
                    if test_period_disc != -1:
                        avg_accuracy = sum(discriminator_accuracy) / len(discriminator_accuracy)
                        print('Average accuracy of discriminator: ', round(avg_accuracy, 5))
                        discriminator_accuracy = []
                    print('Average costs of dicriminator: ', discriminator_cost)
                    print('Average costs of generator: ', generator_cost)

                    generated_images = self._generator.generate(x=x_gen_batch)

                    if final_image_size is not None:
                        generated_images = generated_images.reshape(generated_images.shape[0], *final_image_size)
                    
                    generated_images = np.clip(restore_image_function(generated_images, self._session), 0.0, 255.0).astype(np.uint8)
                    
                    plt.figure(figsize=(20, 20))
                    
                    for z in range(min(batch_size, 100)):
                        plt.subplot(10, 10, z+1)
                        if use_BGR2RGB:
                            plt.imshow(cv2.cvtColor(generated_images[z], cv2.COLOR_BGR2RGB))
                        else:
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


    def fit_ce(
        self, Xtrain, restore_image_function,
        Xgen=None,
        final_image_size=None, 
        optimizer_discriminator=None, 
        optimizer_generator=None, test_period=1,
        epochs=1, test_period_disc=1,
        global_step_disc=None, global_step_gen=None,
        show_images=False,
        label_smoothing=0.9,
        epochs_discriminator=1, epochs_generator=1,
        ):
        """

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
        assert (Xgen is None or len(Xgen) == len(Xtrain))

        batch_size = self._generator_discriminator.get_input_shape()[0]
        n_batches = len(Xtrain) // batch_size

        discriminator_accuracy = []

        y_discriminator = np.zeros((2 * batch_size, *self._discriminator.get_logits_shape()[1:])).astype(np.float32)
        y_discriminator[:batch_size, ...] = label_smoothing

        iterator = None
        try:
            for i in range(epochs):
                print(f'Epochs is {i}')
                if Xgen is None:
                    Xtrain = shuffle(Xtrain)
                else:
                    Xtrain, Xgen = shuffle(Xtrain, Xgen)

                generator_cost = 0.0
                discriminator_cost = 0.0

                iterator = tqdm(range(n_batches))

                for j in iterator:
                    image_batch = Xtrain[j * batch_size : batch_size * (j + 1)]
                    # specific input for generator
                    if Xgen is not None:
                        x_gen_batch = Xgen[j * batch_size : batch_size * (j + 1)]
                    else:
                        x_gen_batch = None

                    generated_images = self._generator.generate(x=x_gen_batch)

                    X_discriminator = np.concatenate([image_batch, generated_images]).astype(np.float32)
                    # Train discriminator
                    info_discriminator = self._discriminator.fit_ce(Xtrain=X_discriminator, Ytrain=y_discriminator,
                                                                    optimizer=optimizer_discriminator,
                                                                    epochs=epochs_discriminator,
                                                                    test_period=test_period_disc,
                                                                    global_step=global_step_disc
                    )
                    if test_period_disc != -1:
                        discriminator_accuracy += info_discriminator[BinaryCETrainingModuleDiscriminator.TRAIN_ACCURACY]
                    disc_cost = info_discriminator[BinaryCETrainingModuleDiscriminator.TRAIN_COSTS]
                    discriminator_cost = moving_average(discriminator_cost, sum(disc_cost) / len(disc_cost), j)

                    # Train generator
                    # if user use l1 or l2 loss
                    if self._generator_discriminator._use_l1 is not None:
                        Xreal = image_batch
                    else:
                        Xreal = None
                        
                    gen_cost = self._generator_discriminator.fit_ce(Xreal=Xreal, Xgen=x_gen_batch,
                                                                    epochs=epochs_generator,
                                                                    optimizer=optimizer_generator,
                                                                    global_step=global_step_gen
                    )[BinaryCETrainingModuleGenerator.TRAIN_COSTS]
                    generator_cost = moving_average(generator_cost, sum(gen_cost) / len(gen_cost), j)

                # close tqdm iterator for out safe
                iterator.close()
                # Validating the network on test data
                if test_period != -1 and i % test_period == 0:
                    print('Test....')
                    if test_period_disc != -1:
                        avg_accuracy = sum(discriminator_accuracy) / len(discriminator_accuracy)
                        print('Average accuracy of discriminator: ', round(avg_accuracy, 5))
                        discriminator_accuracy = []
                    print('Average costs of dicriminator: ', discriminator_cost)
                    print('Average costs of generator: ', generator_cost)

                    if Xgen is not None:
                        gen_input = shuffle(Xgen)[:batch_size]
                    else:
                        gen_input = None
                    generated_images = self._generator.generate(x=gen_input)

                    if final_image_size is not None:
                        generated_images = generated_images.reshape(generated_images.shape[0], *final_image_size)
                    
                    generated_images = np.clip(restore_image_function(generated_images, self._session), 0.0, 255.0).astype(np.uint8)
                    
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
