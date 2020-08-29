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

from makiflow.generators.pipeline.tfr.tfr_map_method import TFRMapMethod, TFRPostMapMethod
from makiflow.models.gans.utils import COLORSPACE_LAB, COLORSPACE_RGB, preprocess
from ..gans_iterator_base import GANsIterator
from .data_preparation import TARGET_X_FNAME, GEN_INPUT_X_FNAME
import tensorflow as tf


class LoadDataMethod(TFRMapMethod):
    def __init__(
            self,
            target_x_shape,
            gen_input_x_shape=None,
            target_x_dtype=tf.float32,
            gen_input_x_dtype=tf.float32
    ):
        """
        Method to load data from records

        Parameters
        ----------
        target_x_shape : list or tuple
            Shape of the loaded image.
        gen_input_x_shape : list or tuple
            Shape of the input image for generator.
            By default set to None, which mean that it does not used further and
            Input for generator is will be some noise.
            Shape of target tensor
        target_x_dtype : tf.dtypes
            Type of target tensor. By default equal to tf.float32
        gen_input_x_dtype : tf.dtypes
            Type of generator input tensor. By default equal to tf.float32
        """
        self.target_x_shape = target_x_shape
        self.gen_input_x_shape = gen_input_x_shape

        self.gen_input_x_dtype = gen_input_x_dtype
        self.target_x_dtype = target_x_dtype

    def read_record(self, serialized_example):
        r_feature_description = {
            TARGET_X_FNAME: tf.io.FixedLenFeature((), tf.string),
        }

        if self.gen_input_x_shape is not None:
            r_feature_description.update({GEN_INPUT_X_FNAME: tf.io.FixedLenFeature((), tf.string)})

        example = tf.io.parse_single_example(serialized_example, r_feature_description)

        # Extract the data from the example
        target_tensor = tf.io.parse_tensor(example[TARGET_X_FNAME], out_type=self.target_x_dtype)
        # Give the data its shape because it doesn't have it right after being extracted
        target_tensor.set_shape(self.target_x_shape)

        # TODO: Change flag image to target_x for all map methods
        output_dict = {
            GANsIterator.IMAGE: target_tensor,
        }

        if self.gen_input_x_shape is not None:
            # Same for generator input, if its used
            gen_input_tensor = tf.io.parse_tensor(example[GEN_INPUT_X_FNAME], out_type=self.gen_input_x_dtype)
            gen_input_tensor.set_shape(self.gen_input_x_shape)

            output_dict.update({GANsIterator.GEN_INPUT: gen_input_tensor})

        return output_dict


class NormalizePostMethod(TFRPostMapMethod):

    NORMALIZE_TARGET_X = 'normalize_target_tensor'
    NORMALIZE_GEN_INPUT_X = 'normalize_gen_input_tensor'

    def __init__(self, divider=127.5,
                 use_caffee_norm=True,
                 use_float64=True,
                 using_for_image_gen=False,
                 using_for_image_gen_only=False):
        """
        Normalizes the tensor by dividing it by the `divider`.
        # TODO: Add docs for other parameters

        Parameters
        ----------
        divider : float or int
            The number to divide the tensor by.
        use_float64 : bool
            Set to True if you want the image to be converted to float64 during normalization.
            It is used for getting more accurate division result during normalization.
        using_for_image_gen : bool
            If true, divider will be used on images for generator.
        """
        super().__init__()
        self.use_float64 = use_float64
        self.use_caffe_norm = use_caffee_norm
        self.using_for_image_gen = using_for_image_gen
        self.using_for_image_gen_only = using_for_image_gen_only

        if use_float64:
            self.divider = tf.constant(divider, dtype=tf.float64)
        else:
            self.divider = tf.constant(divider, dtype=tf.float32)

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)

        target_x = element[GANsIterator.IMAGE]

        if not self.using_for_image_gen_only:

            if self.use_float64:
                # Numpy style
                target_x = tf.cast(target_x, dtype=tf.float64)
                if self.use_caffe_norm:
                    target_x = (target_x - self.divider) / self.divider
                else:
                    target_x = tf.divide(target_x, self.divider, name=NormalizePostMethod.NORMALIZE_TARGET_X)
                target_x = tf.cast(target_x, dtype=tf.float32)
            else:
                if self.use_caffe_norm:
                    target_x = (target_x - self.divider) / self.divider
                else:
                    target_x = tf.divide(target_x, self.divider, name=NormalizePostMethod.NORMALIZE_TARGET_X)
            element[GANsIterator.IMAGE] = target_x

        if self.using_for_image_gen:
            gen_input = element[GANsIterator.GEN_INPUT]
            if self.use_float64:
                # Numpy style
                gen_input = tf.cast(gen_input, dtype=tf.float64)
                if self.use_caffe_norm:
                    gen_input = (gen_input - self.divider) / self.divider
                else:
                    gen_input = tf.divide(gen_input, self.divider, name=NormalizePostMethod.NORMALIZE_GEN_INPUT_X)
                gen_input = tf.cast(gen_input, dtype=tf.float32)
            else:
                if self.use_caffe_norm:
                    gen_input = (gen_input - self.divider) / self.divider
                else:
                    gen_input = tf.divide(gen_input, self.divider, name=NormalizePostMethod.NORMALIZE_GEN_INPUT_X)
            element[GANsIterator.GEN_INPUT] = gen_input

        return element


class RGB2BGRPostMethod(TFRPostMapMethod):

    RGB2BGR_TARGET_X = 'RGB2BGR_tensor'
    RGB2BGR_GEN_INPUT_X = 'BGR2RGB_input'

    def __init__(self, using_for_target_x, using_for_gen_input=False):
        """
        Used for swapping color channels in tensors from RGB to BGR format.
        Parameters
        ----------
        using_for_target_x : bool
            If true, swapping color channels will be used on target x
        using_for_gen_input : bool
            If true, swapping color channels will be used on generator input.
        """
        self.using_for_target_x = using_for_target_x
        self.using_for_gen_input = using_for_gen_input
        super().__init__()

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)

        if self.using_for_target_x:
            # For target tensor
            target_tensor = element[GANsIterator.IMAGE]
            # Swap channels
            element[GANsIterator.IMAGE] = tf.reverse(target_tensor, axis=[-1], name=RGB2BGRPostMethod.RGB2BGR_TARGET_X)

        if self.using_for_target_x:
            # For generator input tensor
            gen_input_tensor = element[GANsIterator.GEN_INPUT]
            # Swap channels
            element[GANsIterator.GEN_INPUT] = tf.reverse(
                gen_input_tensor,
                axis=[-1],
                name=RGB2BGRPostMethod.RGB2BGR_GEN_INPUT_X
            )

        return element


class RGB2LABPostMethod(TFRPostMapMethod):

    def __init__(self):
        """
        Transformate RGB image into LAB image. This post method only for `IMAGE`, not for `GEN_INPUT`.
        NOTICE! After this post method, data will be in [-1, 1] range of values.
        """
        super().__init__()

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)

        target_tensor = element[GANsIterator.IMAGE]
        # Output in range [-1, 1]
        target_tensor = preprocess(target_tensor, COLORSPACE_RGB, COLORSPACE_LAB)
        element[GANsIterator.IMAGE] = target_tensor

        return element


class ResizePostMethod(TFRPostMapMethod):

    def __init__(self,
                 target_tensor_size=None,
                 gen_input_size=None,
                 target_tensor_resize_method=tf.image.ResizeMethod.BILINEAR,
                 gen_input_resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
        """
        Resizes the image and the gen_input accordingly to `target_tensor_size` and `gen_input_size`.

        Parameters
        ----------
        target_tensor_size : list
            List of 2 ints: [image width, image height]. If equal to None, then resize will be not apply
        gen_input_size : list
            List of 2 ints: [gen_input width, gen_input height].  If equal to None, then resize will be not apply
        target_tensor_resize_method : tf.image.ResizeMethod
            Please refer to the TensorFlow documentation for additional information
        gen_input_resize_method : tf.image.ResizeMethod
            Please refer to the TensorFlow documentation for additional information
        """
        super().__init__()
        self.target_tensor_size = target_tensor_size
        self.gen_input_size = gen_input_size
        self.target_tensor_resize_method = target_tensor_resize_method
        self.gen_input_resize_method = gen_input_resize_method

    def read_record(self, serialized_example) -> dict:
        element = self._parent_method.read_record(serialized_example)

        if self.target_tensor_size is not None:
            # For image
            target_tensor = element[GANsIterator.IMAGE]
            target_tensor = tf.image.resize(
                images=target_tensor,
                size=self.target_tensor_size,
                method=self.target_tensor_resize_method
            )

            element[GANsIterator.IMAGE] = target_tensor

        if self.gen_input_size is not None:
            # For generator input
            gen_input_tensor = element[GANsIterator.GEN_INPUT]
            gen_input_tensor = tf.image.resize(
                images=gen_input_tensor,
                size=self.gen_input_size,
                method=self.gen_input_resize_method
            )

            element[GANsIterator.GEN_INPUT] = gen_input_tensor

        return element

