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


from makiflow.models.gans.main_modules import GeneratorDiscriminatorBasic


class PerceptualLossModuleGenerator(GeneratorDiscriminatorBasic):

    NOTIFY_BUILD_PERCEPTUAL_LOSS = "Perceptual loss was built"

    def __init__(self):
        print('create perceptual')
        self._perceptual_loss_vars_are_ready = False

    def is_use_perceptual_loss(self) -> bool:
        """
        Return bool variable which shows whether it is being used perceptual loss or not.
        """
        if not self._perceptual_loss_vars_are_ready:
            return self._perceptual_loss_vars_are_ready

        return self._use_perceptual_loss

    def add_perceptual_loss(self, creation_per_loss, scale_loss=1e-2):
        """
        Add the function that create percetual loss inplace.
        Parameters
        ----------
        creation_per_loss : func
            Function which will create percetual loss.
            This function must have 3 main input: generated_image, target_image, sess.
            Example of function:
                def create_loss(generated_image, target_image, sess):
                    ...
                    ...
                    return percetual_loss
            Where percetual_loss - is tensorflow Tensor
        scale_loss : float
            Scale of the perceptual loss.
        """
        if not self._perceptual_loss_vars_are_ready:
            self._prepare_training_vars()
        self._creation_per_loss = creation_per_loss
        self._scale_per_loss = scale_loss
        self._use_perceptual_loss = True

    def _prepare_training_vars(self):
        if not self._perceptual_loss_vars_are_ready:
            print('prepare was called')
            self._scale_per_loss = 1.0
            self._use_perceptual_loss = False
            self._creation_per_loss = None
            self._perceptual_loss_is_built = False
            self._perceptual_loss_vars_are_ready = True

    def _build_perceptual_loss(self):
        if not self._perceptual_loss_is_built:
            if self._input_real_image is None:
                self._input_real_image = self._discriminator.get_inputs_maki_tensors()[0].get_data_tensor()

            if self._gen_product is None:
                # create output tensor from generator (in train set up)
                self._gen_product = self._return_training_graph_from_certain_output(
                    self._generator.get_outputs_maki_tensors()[0]
                )

            self._perceptual_loss = self._creation_per_loss(self._gen_product,
                                                      self._input_real_image,
                                                      self._session
            ) * self._scale_per_loss
            print(PerceptualLossModuleGenerator.NOTIFY_BUILD_PERCEPTUAL_LOSS)
            self._perceptual_loss_is_built = True

        return self._perceptual_loss
