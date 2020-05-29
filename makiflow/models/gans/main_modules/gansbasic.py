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


from makiflow.base.maki_entities import MakiCore


class GANsBasic(MakiCore):

    NAME = 'name'
    INPUT_S = 'input_s'
    OUTPUT = 'output'

    @staticmethod
    def from_json(path_to_model):
        # TODO
        pass

    def __init__(self, input_s, output, name):
        self.name = str(name)
        graph_tensors = output.get_previous_tensors()
        graph_tensors.update(output.get_self_pair())
        super().__init__(graph_tensors, outputs=[output], inputs=[input_s])
        self._inference_out = self._output_data_tensors[0]

        self._generator_is_set = False
        self._training_vars_are_ready = False

        self._gen_product = None
        self._input_real_image = None

        self._wasserstein_is_built = False

    def predict(self, x):
        return self._session.run(
            self._output_data_tensors[0],
            feed_dict={self._input_data_tensors[0]: x}
        )

    def _get_model_info(self):
        return {
            GANsBasic.NAME: self.name,
            GANsBasic.INPUT_S: self._inputs[0].get_name(),
            GANsBasic.OUTPUT: self._outputs[0].get_name()
        }

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------SETTING UP TRAINING-------------------------------------

    def get_layers_names(self) -> dict:
        """
        Returns
        -------
        dict of MakiTensors
            All the MakiTensors that appear earlier in the computational graph.
            The dictionary contains pairs: { name of the tensor: MakiTensor }.
        """
        return self._graph_tensors

    def get_logits_shape(self):
        return self._outputs[0].get_shape()

    def get_outputs_maki_tensors(self) -> list:
        return self._outputs

    def get_input_shape(self):
        shape = self._inputs[0].get_shape()
        return shape

    def get_inputs_maki_tensors(self) -> list:
        return self._inputs

