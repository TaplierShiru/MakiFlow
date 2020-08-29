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

from makiflow.generators.pipeline.map_method import MapMethod, PostMapMethod
import tensorflow as tf
from makiflow.models.gans.utils import COLORSPACE_LAB, COLORSPACE_RGB, preprocess
from ..gans_iterator_base import GANsIterator


class LoadDataMethodSimpleForm(MapMethod):
    def __init__(self, image_shape, gen_input_shape=None):
        """
        Method to load data. It should be first method in map methods.
        This is a simple form of load data method, i. e. images loaded from certain folder, which are not tfrecords
        That can be can faster for some cases.
        So, this method can't load something except images (PNG, JPG, BMP and etc...).
        For loading other type of data refer to class LoadDataMethod, which is load tfrecords.

        Parameters
        ----------
        image_shape : list or tuple
            Shape of the loaded image.
        gen_input_shape : list or tuple
            Shape of the input image for generator.
            By default set to None, which mean that input for generator is some noise.
        """
        self.image_shape = image_shape
        self.gen_input_shape = gen_input_shape

    def load_data(self, data_paths):
        # for image
        img_file = tf.read_file(data_paths[GANsIterator.IMAGE])
        img = tf.image.decode_image(img_file)
        img.set_shape(self.image_shape)
        img = tf.cast(img, dtype=tf.float32)
        # for generator
        if self.gen_input_shape is not None:
            gen_input_file = tf.read_file(data_paths[GANsIterator.GEN_INPUT])
            gen_input = tf.image.decode_image(gen_input_file)
            gen_input.set_shape(self.gen_input_shape)
            gen_input = tf.cast(gen_input, dtype=tf.int32)

            return {
                GANsIterator.IMAGE: img,
                GANsIterator.GEN_INPUT: gen_input
            }

        return {
            GANsIterator.IMAGE: img
        }


class ResizePostMethod(PostMapMethod):
    def __init__(self, image_size=None, gen_input_size=None, image_resize_method=tf.image.ResizeMethod.BILINEAR,
                 gen_input_resize_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR):
        """
        Resizes the image and the gen_input accordingly to `image_size` and `gen_input_size`.
        Parameters
        ----------
        image_size : list
            List of 2 ints: [image width, image height].
        gen_input_size : list
            List of 2 ints: [gen_input width, gen_input height].
            By default set to None, which mean that input for generator is some noise.
            So resize will be used only for image.
            If you just do not want to resize images for generator, set it to [].
        image_resize_method : tf.image.ResizeMethod
            Please refer to the TensorFlow documentation for additional info.
        gen_input_resize_method : tf.image.ResizeMethod
            Please refer to the TensorFlow documentation for additional info.
        """
        super().__init__()
        self.image_size = image_size
        self.gen_input_size = gen_input_size
        self.image_resize_method = image_resize_method
        self.gen_input_resize_method = gen_input_resize_method

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        # for image
        img = element[GANsIterator.IMAGE]
        if self.image_size is not None:
            img = tf.image.resize(images=img, size=self.image_size, method=self.image_resize_method)
        # for generator
        if self.gen_input_size is not None:
            gen_input = element[GANsIterator.GEN_INPUT]
            if len(self.gen_input_size) != 0:
                gen_input = tf.image.resize(images=gen_input,
                                            size=self.gen_input_size, method=self.gen_input_resize_method)
            return {
                GANsIterator.IMAGE: img,
                GANsIterator.GEN_INPUT: gen_input
            }

        return {
            GANsIterator.IMAGE: img
        }


class RGB2LABPostMethod(PostMapMethod):
    def __init__(self):
        """
        Transformate RGB image into LAB image. This post method only for `IMAGE`, not for `GEN_INPUT`.
        NOTICE! After this post method, data will be in [-1, 1] range of values.
        """
        super().__init__()

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        img = element[GANsIterator.IMAGE]
        # output in range [-1, 1]
        img = preprocess(img, COLORSPACE_RGB, COLORSPACE_LAB)
        element[GANsIterator.IMAGE] = img
        return element


class NormalizePostMethod(PostMapMethod):
    def __init__(self, divider=127.5, use_caffee_norm=True, use_float64=True, using_for_image_gen=False,
                 using_for_image_gen_only=False):
        """
        Normalizes the image by dividing it by the `divider`.
        Parameters
        ----------
        divider : float or int
            The number to divide the image by.
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

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        img = element[GANsIterator.IMAGE]
        if not self.using_for_image_gen_only:
            if self.use_float64:
                img = tf.cast(img, dtype=tf.float64)
                if self.use_caffe_norm:
                    img = (img - self.divider) / self.divider
                else:
                    img = tf.divide(img, self.divider, name='normalizing_image')
                img = tf.cast(img, dtype=tf.float32)
            else:
                if self.use_caffe_norm:
                    img = (img - self.divider) / self.divider
                else:
                    img = tf.divide(img, self.divider, name='normalizing_image')
            element[GANsIterator.IMAGE] = img

        if self.using_for_image_gen:
            gen_input = element[GANsIterator.GEN_INPUT]
            if self.use_float64:
                gen_input = tf.cast(gen_input, dtype=tf.float64)
                if self.use_caffe_norm:
                    gen_input = (gen_input - self.divider) / self.divider
                else:
                    gen_input = tf.divide(gen_input, self.divider, name='normalizing_gen_input')
                gen_input = tf.cast(gen_input, dtype=tf.float32)
            else:
                if self.use_caffe_norm:
                    gen_input = (gen_input - self.divider) / self.divider
                else:
                    gen_input = tf.divide(gen_input, self.divider, name='normalizing_gen_input')
            element[GANsIterator.GEN_INPUT] = gen_input

        return element


class RGB2BGRPostMethod(PostMapMethod):
    def __init__(self, using_for_image_gen=False):
        """
        Used for swapping color channels in images from RGB to BGR format.

        Parameters
        ----------
        using_for_image_gen : bool
            If true, swapping color channels will be used on images for generator.
        """
        self.using_for_image_gen = using_for_image_gen
        super().__init__()

    def load_data(self, data_paths):
        element = self._parent_method.load_data(data_paths)
        # for image
        img = element[GANsIterator.IMAGE]
        # Swap channels
        element[GANsIterator.IMAGE] = tf.reverse(img, axis=[-1], name='RGB2BGR_image')
        # for generator
        if self.using_for_image_gen:
            gen_input = element[GANsIterator.GEN_INPUT]
            # Swap channels
            element[GANsIterator.GEN_INPUT] = tf.reverse(gen_input, axis=[-1], name='RGB2BGR_gen_input')
        return element
