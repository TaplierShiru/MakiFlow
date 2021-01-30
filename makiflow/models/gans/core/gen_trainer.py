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

from makiflow.core import MakiTrainer
import tensorflow as tf
from abc import ABC, abstractmethod

from makiflow.models.regressor import Regressor
from makiflow.core import MakiModel


class GeneratorTrainer(MakiTrainer, ABC):
    WEIGHT_MAP = 'WEIGHT_MAP'
    LABELS = 'LABELS'

    def __init__(self, discriminator: Regressor, model: MakiModel):
        # We must connect graph of the model with discriminator
        # For further train stuff
        # Layers from discriminator must be frozen
        self._discriminator = discriminator

        # model - generator in our case

        gen_in_x = model.get_inputs()[0]
        gen_output_x = model.get_outputs()[0]

        disc_in_x = discriminator.get_inputs()[0]
        disc_output_x = discriminator.get_outputs()[0]

        connected_maki_tensors_output = self._connect_generator_disc_graph(gen_output_x, disc_in_x, disc_output_x)[0]
        # TODO: Merged graph create another placeholder variable, which is needed in furhter usage
        # TODO: (?), change vars name or ...
        connected_model = Regressor(in_x=[gen_in_x], out_x=[connected_maki_tensors_output], name='MergedGraph')
        # TODO: Debuf this pease
        super().__init__(
            model=connected_model,
            train_inputs=[gen_in_x],
            label_tensors=None
        )
        # Set layers from generator in `generator_discriminator` as untrained
        untrain = []
        generator_names = list(model.get_layers().keys())
        for name in discriminator.get_layers().keys():
            if name not in generator_names:
                untrain.append((name, False))

        self.set_layers_trainable(untrain)

    def _connect_generator_disc_graph(self, output_generator, input_discriminator, output_discriminator):
        """
        Connect two graph in one, which is used for training
        """
        stop_point = input_discriminator.get_name()
        outputs = [output_discriminator]

        used = {}
        layer_names = []

        def find_names(from_):
            if used.get(from_.get_name()) is None:
                layer = from_.get_parent_layer()
                used[layer.get_name()] = True
                # Check if we at the beginning of the computational graph, i.e. InputLayer
                if len(from_.get_parent_tensor_names()) != 0:

                    for elem in from_.get_parent_tensor_names():
                        if stop_point == elem:
                            layer_names.append(from_.get_name())

                    for elem in from_.get_parent_tensors():
                        find_names(elem)

        # Create list of the layers names from discriminator
        for output in outputs:
            find_names(output)
            # Contains pairs {layer_name: tensor}, where `tensor` is output
        # tensor of layer called `layer_name`
        output_tensors = {}
        used = {}

        def create_tensor(from_):
            if used.get(from_.get_name()) is None:
                change = False

                layer = from_.get_parent_layer()
                used[layer.get_name()] = True
                X = from_
                takes = []
                if from_.get_name() in layer_names:
                    prev = [output_generator.get_name()]
                    change = True
                else:
                    prev = from_.get_parent_tensor_names()
                # Check if we at the beginning of the computational graph, i.e. InputLayer
                if len(prev) != 0:
                    if change:
                        takes += [create_tensor(output_generator)]
                    else:
                        for elem in from_.get_parent_tensors():
                            takes += [create_tensor(elem)]

                    X = layer(takes[0] if len(takes) == 1 else takes)

                output_tensors[layer.get_name()] = X
                return X
            else:
                return output_tensors[from_.get_name()]

        training_outputs = []
        for output in outputs:
            training_outputs += [create_tensor(output)]

        return training_outputs

    def _init(self):
        super()._init()
        self._use_weight_mask = False
        logits_makitensor = super().get_model().get_logits()
        self._logits_names = [logits_mk_single.get_name() for logits_mk_single in logits_makitensor]
        self._labels = super().get_label_tensors()

    # noinspection PyAttributeOutsideInit
    def set_loss_sources(self, source_names):
        self._logits_names = source_names

    def get_labels(self):
        return self._labels

    def get_logits(self):
        logits = []
        for name in self._logits_names:
            logits.append(super().get_traingraph_tensor(name))
        return logits

    @abstractmethod
    def _build_local_loss(self, prediction, label):
        pass

    def _build_loss(self):
        losses = []
        for name in self._logits_names:
            prediction = super().get_traingraph_tensor(name)
            label = self.get_labels()[name]
            losses.append(self._build_local_loss(prediction, label))
            super().track_loss(losses[-1], name)
        return tf.add_n([0.0, *losses], name='total_loss')

    def _setup_label_placeholders(self):
        print('setup')
        logits = super().get_model().get_logits()
        batch_size = super().get_batch_size()
        label_tensors = {}
        for l_single in logits:
            print(type(l_single))
            print(f'create: label_{l_single.get_name()}  layers: {l_single.get_name()}')
            label_tensors[l_single.get_name()] = tf.placeholder(
                dtype='float32', shape=[batch_size, *l_single.get_shape()[1:]], name=f'label_{l_single.get_name()}'
            )
        return label_tensors

    def get_label_feed_dict_config(self):
        labels = super().get_label_tensors()
        label_feed_dict_config = {}
        for i, t in enumerate(labels.values()):
            label_feed_dict_config[t] = i
        return label_feed_dict_config

