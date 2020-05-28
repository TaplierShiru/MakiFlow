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
import numpy as np

# TODO: Add comments into model

class Generator(GANsBasic):

    def __init__(self, input_s, output, name, use_noise_as_input=True):
        self._use_noise_as_input = use_noise_as_input
        super().__init__(input_s=input_s, output=output, name=name)

    def generate(self, x=None):
        if x is None:
            x = self.get_noise()
        return self._session.run(
            self._output_data_tensors[0],
            feed_dict={self._input_data_tensors[0]: x}
        )

    def get_noise(self, size=None):
        if not self._use_noise_as_input:
            return None

        if size is None:
            x = np.random.normal(0, 1, size=super().get_input_shape())
        else:
            x = np.random.normal(0, 1, size=size)
        return x.astype(np.float32)
