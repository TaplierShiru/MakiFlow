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
import traceback

import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
import numpy as np

from makiflow.core.training.utils import moving_average
from makiflow.models.gans.core import DiscriminatorTrainer, GeneratorTrainer
from makiflow.models.gans.generator_model import GeneratorModel
from makiflow.models.gans.utils import visualise_sheets_of_images


class GansController:
    SIMPLE_GAN = 'SimpleGAN'

    def __init__(
            self,
            generator: GeneratorModel,
            discriminator: DiscriminatorTrainer,
            generator_discriminator: GeneratorTrainer):
        """
        Create simple GANs model for training.

        Parameters
        ----------
        generator : Generator
            Model of the generator.
        discriminator : Discriminator
            Model of the discriminator.
        generator_discriminator : GeneratorDiscriminator
            Combined model of the discriminator and generator, which is used for training.

        """
        self._discriminator = discriminator
        self._generator = generator
        self._session = None
        self._generator_discriminator = generator_discriminator

        # Variable for generators (pipeline stuff)
        self._generator_is_set = None
        self._gen_in_target = None
        self._gen_in_input = None

    def set_session(self, sess: tf.Session):
        """
        Set session in generator, discriminator, generator and discriminator (combined model) models.

        Parameters
        ----------
        sess : tf.Session
        """
        self._session = sess

        self._generator_discriminator._model._session = sess
        self._discriminator.get_model().set_session(sess)
        self._generator.set_session(sess)

    def set_python_generator(self, gen_in_target, gen_in_input):
        """
        Set generators for training.

        Parameters
        ----------
        gen_in_target : InputGenLayer
            Is target images for generator.
        gen_in_input : InputGenLayer
            Is input images for generator. If your model use noise as input, just set to None.
        """
        self._generator_is_set = False
        self._gen_in_target = gen_in_target
        self._gen_in_input = gen_in_input

    def set_generator(self, gen_in_target, gen_in_input):
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
            self, iterations,
            restore_image_function,
            final_image_size=None,
            optimizer_discriminator=None,
            optimizer_generator=None,
            test_period=1,
            epochs=1,
            test_period_disc=1,
            global_step_disc=None,
            global_step_gen=None,
            show_images=False,
            label_smoothing=0.9,
            use_BGR2RGB=False,
            pre_download=20,
            epochs_discriminator=1,
            epochs_generator=1,):
        """
        Start training of the SimpleGAN with pipeline.
        TODO: Write full docs.
        Parameters
        ----------
        iterations : int
        restore_image_function : python function
            Example: `def restore_image_function(img, sess): ... `.
        final_image_size : list
            Example: [32, 32, 3].
        optimizer_discriminator : tf.optimizer
        optimizer_generator : tf.optimizer
        test_period : int
        epochs : int
        global_step_gen : tf.Variable
        global_step_disc : tf.Variable
        test_period_disc : int
            Prints period of accuracy of the discriminator. Set it to -1, to speed up training,
            otherwise it can slows down the training.
        show_images : bool
        label_smoothing : float
        use_BGR2RGB : bool
        pre_download : int
        epochs_generator : int
        epochs_discriminator : int

        Returns
        -------
            python dictionary
                Dictionary with all testing data(train error, train cost, test error, test cost)
                for each test period.
        """

        assert (optimizer_discriminator is not None)
        assert (optimizer_generator is not None)
        assert (self._session is not None)
        assert (self._generator_is_set is not None), "Provide some generator for data\n" + \
                                                     "See methods: `set_python_generator` and `set_generator` "

        batch_size = self._generator_discriminator.get_batch_size()
        discriminator_accuracy = []
        iterations = int(iterations / pre_download) * pre_download
        y_discriminator = np.zeros((2 * batch_size, *self._discriminator.get_logits()[0].get_shape()[1:])).astype(np.float32)
        y_discriminator[:batch_size, ...] = label_smoothing
        y_generator = np.ones(self._generator_discriminator.get_logits()[0].get_shape()).astype(np.float32)

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
                    if current_batch_preload == pre_download or len(image_batch_preload) == 0:
                        current_batch_preload = 0
                        x_gen_batch_preload = []
                        image_batch_preload = []

                        for _ in range(pre_download):
                            if self._gen_in_input is not None:
                                # Take images/data for generator (as input) and for discriminator
                                if self._generator_is_set:
                                    # if we separate this, generator will provide different images
                                    x_gen_batch_single, image_batch_single = self._session.run([
                                        self._gen_in_input.get_data_tensor(),
                                        self._gen_in_target.get_data_tensor()]
                                    )
                                else:
                                    x_gen_batch_single, image_batch_single = (
                                        next(self._gen_in_input),
                                        next(self._gen_in_target)
                                    )
                                x_gen_batch_preload.append(x_gen_batch_single)
                            else:
                                # Only image for discriminator
                                if self._generator_is_set:
                                    image_batch_single = self._session.run(self._gen_in_target.get_data_tensor())
                                else:
                                    image_batch_single = next(self._gen_in_target)

                            image_batch_preload.append(image_batch_single)

                    image_batch = image_batch_preload[current_batch_preload]
                    # specific input for generator
                    if self._gen_in_input is not None:
                        x_gen_batch = x_gen_batch_preload[current_batch_preload]
                    else:
                        x_gen_batch = self._generator.get_noise()

                    current_batch_preload += 1

                    generated_images = self._generator.generate(x=x_gen_batch)

                    x_discriminator = np.concatenate([image_batch, generated_images]).astype(np.float32)

                    def wrapper_gen(x, y):
                        counter = 0
                        size = len(x) // batch_size
                        while True:
                            batch_x = x[counter*batch_size: batch_size*(counter+1)]
                            batch_y = y[counter*batch_size: batch_size*(counter+1)]
                            yield [batch_x], [batch_y]
                            counter += 1
                            if counter != 0 and size == counter:
                                counter = 0

                    print('x_disc: ', x_discriminator.shape)
                    gen_disc_data = wrapper_gen(x_discriminator, y_discriminator)

                    print('x_disc next1: ', next(gen_disc_data)[0][0].shape)
                    print('x_disc next2: ', next(gen_disc_data)[0][0].shape)
                    # Train discriminator
                    # TODO: Do test stuff with discriminator according to `test_period_disc`, i.e. call evaluate
                    info_discriminator = self._discriminator.fit_generator(
                        generator=gen_disc_data,
                        optimizer=optimizer_discriminator,
                        epochs=epochs_discriminator,
                        global_step=global_step_disc,
                        iter=2,
                    )

                    #if test_period_disc != -1:
                    #    discriminator_accuracy += info_discriminator[BinaryCETrainingModuleDiscriminator.TRAIN_ACCURACY]
                    # TODO: Store disciminator accuracy for futher print of it

                    # For now, just take first value of loss
                    disc_cost = info_discriminator[list(info_discriminator.keys())[0]]
                    discriminator_cost = moving_average(discriminator_cost, sum(disc_cost) / len(disc_cost), j)

                    # Train generator
                    data_for_gen_x = [x_gen_batch]
                    # if user use l1 or l2 loss
                    if self._gen_in_input is None:
                        data_for_gen_x += [image_batch]

                    gen_gen_data = wrapper_gen(data_for_gen_x, y_generator)
                    info_gen = self._generator_discriminator.fit_generator(
                        generator=gen_gen_data,
                        epochs=epochs_generator,
                        optimizer=optimizer_generator,
                        global_step=global_step_gen
                    )
                    gen_cost = info_gen[list(info_gen.keys())[0]]
                    generator_cost = moving_average(generator_cost, sum(gen_cost) / len(gen_cost), j)

                # close tqdm iterator for out safe
                iterator.close()

                # Validating the network on test data
                if test_period != -1 and i % test_period == 0:
                    # TODO: Write additional tools for printing stuff
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

                    generated_images = np.clip(
                        restore_image_function(generated_images, self._session),
                        0.0,
                        255.0
                    ).astype(np.uint8)

                    visualise_sheets_of_images(
                        generated_images,
                        prefix_name=GansController.SIMPLE_GAN,
                        unique_index=i,
                        show_images=show_images,
                        use_BGR2RGB=use_BGR2RGB
                    )

        except Exception as ex:
            print(traceback.print_exc())
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()

