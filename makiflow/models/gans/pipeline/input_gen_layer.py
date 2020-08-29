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


from makiflow.generators.pipeline.gen_base import GenLayer

# TODO: Add docs


class InputGenLayer(GenLayer):
    def __init__(
            self, iterator, name, input_tensor_name: str
    ):
        self.iterator = iterator
        self.input_tensor_name = input_tensor_name
        super().__init__(
            name=name,
            input_tensor=self.iterator.get_iterator()[input_tensor_name]
        )

    def get_iterator(self):
        return self.iterator.get_iterator()[self.input_tensor_name]
