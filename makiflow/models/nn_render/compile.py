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

from .training_modules import AbsTrainingModule, MseTrainingModule, MaskedAbsTrainingModule, \
    MaskedMseTrainingModule, PerceptualTrainingModule, AdversarialTrainingModule

from makiflow.models.gans.simple_gan import Discriminator
from makiflow.models.gans.training_modules.binary_ce_loss import BinaryCETrainingModuleDiscriminator
from makiflow.models.gans.utils import visualise_sheets_of_images
from makiflow.generators.nn_render import NNRIterator
import tensorflow as tf
import numpy as np
from makiflow.generators.pipeline.tfr.tfr_gen_layers import InputGenLayerV2
from tqdm import tqdm
from makiflow.models.common.utils import moving_average
from sklearn.utils import shuffle


class NeuralRender(
    AbsTrainingModule,
    MseTrainingModule,
    MaskedAbsTrainingModule,
    MaskedMseTrainingModule,
    PerceptualTrainingModule
):
    pass


class NeuralRenderAdversarialTrainer(AdversarialTrainingModule):
    """
    Train Neural render with additional model discriminator

    """
    pass


class NeuralRenderAdversarialControler:
    """
    Control training of the neural render and discriminator

    """
    ADVERSARIAL_CONTROLER = 'ADVERSARIAL_CONTROLER'

    def __init__(self,
                 generator: NeuralRender,
                 discriminator: Discriminator,
                 generator_discriminator: NeuralRenderAdversarialTrainer):
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
        # Set layers from generator in `generator_discriminator` as untrained
        untrain = []
        generator_names = list(generator.get_layers_names().keys())
        for name in generator_discriminator.get_layers_names().keys():
            if name not in generator_names:
                untrain.append((name, False))

        generator_discriminator.set_layers_trainable(untrain)
        self._session = None
        self._generator_discriminator = generator_discriminator
        # Variable for generators (pipeline stuff)
        self._generator_is_set = False
        self._gen_in_target = None
        self._gen_in_input = None
        self._gen_in_mask = None

    def set_session(self, sess: tf.Session):
        """
        Set session in generator, discriminator, generator and discriminator (combined model) models.

        Parameters
        ----------
        sess : tf.Session
        """
        self._session = sess

        self._generator_discriminator.set_session(sess)
        self._discriminator._session = sess
        self._generator._session = sess

    def set_generator(self, gen: InputGenLayerV2):
        """
        Set generators for training.

        Parameters
        ----------
        gen : InputGenLayerV2
        """
        self._generator_is_set = True
        self._gen_in_target = gen.get_iterator()[NNRIterator.IMAGE]
        self._gen_in_input = gen.get_iterator()[NNRIterator.UVMAP]
        if self._generator_discriminator._use_mask:
            self._gen_in_mask = gen.get_iterator()[NNRIterator.BIN_MASK]


    def genfit_ce(
            self, iterations, restore_image_function,
            optimizer_discriminator=None,
            optimizer_generator=None, test_period=1,
            epochs=1, test_period_disc=1,
            global_step_disc=None, global_step_gen=None,
            show_images=False,
            label_smoothing=0.9,
            use_BGR2RGB=False,
            pre_download=20,
            epochs_discriminator=1, epochs_generator=1,
    ):
        """
        Start training of the SimpleGAN with pipeline.
        TODO: Write full docs.
        Parameters
        ----------
        iterations : int
        restore_image_function : python function
            Example: `def restore_image_function(img, sess): ... `.
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
        assert (self._generator_is_set)

        batch_size = self._generator_discriminator.get_input_shape()[0]

        discriminator_accuracy = []
        iterations = int(iterations / pre_download) * pre_download
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
                mask_batch_preload = []

                x_gen_batch = None
                current_batch_preload = 0

                for j in iterator:
                    # specific input for generator
                    # this implementation very slow down training, example below more faster
                    # TODO: Delete this code if second type is satisfies for us
                    """
                    if self._gen_in_input is not None:
                        # if we separate this, generator will provide different images
                        x_gen_batch, image_batch = self._session.run([self._gen_in_input.get_data_tensor(),
                                                                      self._gen_in_target.get_data_tensor()]
                        )
                    else:
                        image_batch = self._session.run(self._gen_in_target.get_data_tensor())
                    """
                    # second type of the pipeline usage, which more faster
                    # if we preload more than 1 batch
                    if current_batch_preload == pre_download or len(image_batch_preload) == 0:
                        current_batch_preload = 0
                        x_gen_batch_preload = []
                        image_batch_preload = []
                        mask_batch_preload = []

                        for _ in range(pre_download):
                            if self._gen_in_input is not None:
                                # if we separate this, generator will provide different images
                                if self._gen_in_mask is None:
                                    x_gen_batch_single, image_batch_single = self._session.run([
                                        self._gen_in_input,
                                        self._gen_in_target
                                        ]
                                    )
                                else:
                                    x_gen_batch_single, image_batch_single, mask_single = self._session.run([
                                        self._gen_in_input,
                                        self._gen_in_target,
                                        self._gen_in_mask
                                        ]
                                    )
                                    mask_batch_preload.append(mask_single)

                                x_gen_batch_preload.append(x_gen_batch_single)
                            else:
                                image_batch_single = self._session.run(self._gen_in_target)
                            image_batch_preload.append(image_batch_single)

                    image_batch = image_batch_preload[current_batch_preload]
                    # specific input for generator
                    if self._gen_in_input is not None:
                        x_gen_batch = x_gen_batch_preload[current_batch_preload]
                    else:
                        x_gen_batch = None

                    current_batch_preload += 1

                    generated_images = self._generator.predict(x=x_gen_batch)

                    x_discriminator = np.concatenate([image_batch, generated_images]).astype(np.float32)
                    # Train discriminator
                    info_discriminator = self._discriminator.fit_ce(
                        Xtrain=x_discriminator,
                        Ytrain=y_discriminator,
                        optimizer=optimizer_discriminator,
                        epochs=epochs_discriminator,
                        test_period=test_period_disc,
                        global_step=global_step_disc
                    )
                    if test_period_disc != -1:
                        discriminator_accuracy += info_discriminator[
                            BinaryCETrainingModuleDiscriminator.TRAIN_ACCURACY]
                    disc_cost = info_discriminator[BinaryCETrainingModuleDiscriminator.TRAIN_COSTS]
                    discriminator_cost = moving_average(discriminator_cost, sum(disc_cost) / len(disc_cost), j)

                    # Train generator
                    gen_cost = self._generator_discriminator.fit_ce(
                        Xreal=image_batch, Xgen=x_gen_batch, Xmask=mask_batch_preload,
                        epochs=epochs_generator,
                        optimizer=optimizer_generator,
                        global_step=global_step_gen
                    )[AdversarialTrainingModule.TRAIN_COSTS]
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

                    x_gen_batch = shuffle(x_gen_batch)
                    generated_images = self._generator.predict(x=x_gen_batch)

                    generated_images = np.clip(
                        restore_image_function(generated_images, self._session),
                        0.0,
                        255.0
                    ).astype(np.uint8)

                    visualise_sheets_of_images(
                        generated_images,
                        prefix_name=NeuralRenderAdversarialControler.ADVERSARIAL_CONTROLER,
                        unique_index=i,
                        show_images=show_images,
                        use_BGR2RGB=use_BGR2RGB
                    )

        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()

    def fit_ce(
            self, Xtrain, restore_image_function,
            Xgen=None,
            Xmask=None,
            optimizer_discriminator=None,
            optimizer_generator=None, test_period=1,
            epochs=1, test_period_disc=1,
            global_step_disc=None, global_step_gen=None,
            show_images=False,
            use_BGR2RGB=False,
            label_smoothing=0.9,
            epochs_discriminator=1, epochs_generator=1,
    ):
        """
        Start training of the SimpleGAN.
        TODO: Write full docs.
        TODO: Add support mask, `Xmask` input
        Parameters
        ----------
        Xtrain : list or np.ndarray
        Xgen : list or np.ndarray
        restore_image_function : python function
            Example: `def restore_image_function(img, sess): ... `.
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
                    image_batch = Xtrain[j * batch_size: batch_size * (j + 1)]
                    # specific input for generator
                    if Xgen is not None:
                        x_gen_batch = Xgen[j * batch_size: batch_size * (j + 1)]
                    else:
                        x_gen_batch = None

                    generated_images = self._generator.generate(x=x_gen_batch)

                    x_discriminator = np.concatenate([image_batch, generated_images]).astype(np.float32)

                    # Train discriminator
                    info_discriminator = self._discriminator.fit_ce(
                        Xtrain=x_discriminator, Ytrain=y_discriminator,
                        optimizer=optimizer_discriminator,
                        epochs=epochs_discriminator,
                        test_period=test_period_disc,
                        global_step=global_step_disc
                    )

                    if test_period_disc != -1:
                        discriminator_accuracy += info_discriminator[
                            BinaryCETrainingModuleDiscriminator.TRAIN_ACCURACY]
                    disc_cost = info_discriminator[BinaryCETrainingModuleDiscriminator.TRAIN_COSTS]
                    discriminator_cost = moving_average(discriminator_cost, sum(disc_cost) / len(disc_cost), j)

                    # Train generator
                    gen_cost = self._generator_discriminator.fit_ce(
                        Xreal=image_batch, Xgen=x_gen_batch,
                        epochs=epochs_generator,
                        optimizer=optimizer_generator,
                        global_step=global_step_gen
                    )[AdversarialTrainingModule.TRAIN_COSTS]
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

                    gen_input = shuffle(Xgen)[:batch_size]
                    generated_images = self._generator.predict(x=gen_input)

                    # Give session of there is some special normalization via the TensorFlow
                    generated_images = np.clip(
                        restore_image_function(generated_images, self._session),
                        0.0,
                        255.0
                    ).astype(np.uint8)

                    visualise_sheets_of_images(
                        generated_images,
                        prefix_name=NeuralRenderAdversarialControler.ADVERSARIAL_CONTROLER,
                        unique_index=i,
                        show_images=show_images,
                        use_BGR2RGB=use_BGR2RGB
                    )

        except Exception as ex:
            print(ex)
            print('type of error is ', type(ex))
        finally:
            if iterator is not None:
                iterator.close()
