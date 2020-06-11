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


from .main_modules import GANsBasic
from makiflow.base import InputMakiLayer, MakiTensor
import numpy as np


class Generator(GANsBasic):

    def __init__(self,
                 input_s: InputMakiLayer,
                 output: MakiTensor,
                 name: str,
                 use_noise_as_input=True,
                 custom_noise_function=None):
        """
        Create model of the Generator.

        Parameters
        ----------
        input_s : InputMakiLayer
            Input layer for this model.
        output : MakiTensor
            Output tensor for this model.
        name : str
            Name of this model.
        use_noise_as_input : bool
            If equal to true as input will be noise ( `np.random.normal` in range `(0, 1)` ),
            otherwise as input will be some images.
        custom_noise_function : func
            Function that return certain noise according to input size,
            i. e. this function should have input parameter `size`. By default will be used function from example below.
            Example:
            def get_noise_example(size=None):
                return np.random.normal(0, 1, size=size)
        """
        if use_noise_as_input:
            if custom_noise_function is not None:
                self._custom_noise_function = custom_noise_function
            else:
                def get_noise_example(size=None):
                    return np.random.normal(0, 1, size=size)
                self._custom_noise_function = get_noise_example
        else:
            self._custom_noise_function = None
        self._use_noise_as_input = use_noise_as_input
        super().__init__(input_s=input_s, output=output, name=name)

    def generate(self, x=None):
        """
        Generate image

        Parameters
        ----------
        x : np.ndarray of list
            Input array for this model, by default equal to `None` which mean that input it's noise,
            otherwise it can be list of images or some other form of noise

        Returns
        ----------
        same type as input
            Generated images
        """
        if x is None:
            x = self.get_noise()
        return self._session.run(
            self._output_data_tensors[0],
            feed_dict={self._input_data_tensors[0]: x}
        )

    def get_noise(self, size=None):
        """
        Get noise of these model with certain size

        Parameters
        ----------
        size : tuple or list
            Size of the noise array.

        Returns
        ----------
        np.ndarray
            Noise with certain size according to `size` parameter.
            NOTICE! If `use_noise_as_input` was set as `False` as output will be `None` value.
        """
        if not self._use_noise_as_input:
            return None

        if size is None:
            x = self._custom_noise_function(size=super().get_input_shape())
        else:
            x = self._custom_noise_function(size=size)
        return x.astype(np.float32)
