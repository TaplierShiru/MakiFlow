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

import numpy as np
import tensorflow as tf

from makiflow.base.maki_entities.maki_layer import MakiRestorable
from makiflow.layers.activation_converter import ActivationConverter
from makiflow.layers.sf_layer import SimpleForwardLayer
from makiflow.base import BatchNormBaseLayer
from makiflow.layers.utils import InitConvKernel, InitDenseMat



def spectral_normed_weight(w, name,
    u, 
    num_iters=1, # For Power iteration method, usually num_iters = 1 will be enough
    update_collection=None, 
    with_sigma=False # Estimated Spectral Norm
    ):

    w_shape = w.shape.as_list()
    w_new_shape = [ np.prod(w_shape[:-1]), w_shape[-1] ]
    w_reshaped = tf.reshape(w, w_new_shape, name='w_reshaped' + name)
    
    # power iteration
    u_ = u
    for _ in range(num_iters):
        # ( w_new_shape[1], w_new_shape[0] ) * ( w_new_shape[0], 1 ) -> ( w_new_shape[1], 1 )
        v_ = _l2normalize(tf.matmul(tf.transpose(w_reshaped), u_),  name) 
        # ( w_new_shape[0], w_new_shape[1] ) * ( w_new_shape[1], 1 ) -> ( w_new_shape[0], 1 )
        u_ = _l2normalize(tf.matmul(w_reshaped, v_),  name)

    u_final = tf.identity(u_, name='u_final' + name) # ( w_new_shape[0], 1 )
    v_final = tf.identity(v_, name='v_final' + name) # ( w_new_shape[1], 1 )

    u_final = tf.stop_gradient(u_final)
    v_final = tf.stop_gradient(v_final)

    sigma = tf.matmul(tf.matmul(tf.transpose(u_final), w_reshaped), v_final, name="est_sigma" + name)

    u.assign(u_final)

    sigma = tf.identity(sigma)
    w_bar = tf.identity(w / sigma, 'w_bar' + name)

    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar

def _l2normalize(v, name, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)



class ConvLayerSpectral(SimpleForwardLayer):
    TYPE = 'ConvLayer'
    SHAPE = 'shape'
    STRIDE = 'stride'
    PADDING = 'padding'
    ACTIVATION = 'activation'
    USE_BIAS = 'use_bias'
    INIT_TYPE = 'init_type'

    def __init__(self, kw, kh, in_f, out_f, name, stride=1, padding='SAME', activation=tf.nn.relu,
                 kernel_initializer=InitConvKernel.HE, use_bias=True, regularize_bias=False, W=None, b=None):
        """
        Parameters
        ----------
        kw : int
            Kernel width.
        kh : int
            Kernel height.
        in_f : int
            Number of input feature maps. Treat as color channels if this layer
            is first one.
        out_f : int
            Number of output feature maps (number of filters).
        stride : int
            Defines the stride of the convolution.
        padding : str
            Padding mode for convolution operation. Options: 'SAME', 'VALID' (case sensitive).
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        b : numpy array
            Bias' weights. This value is used for the bias initialization with pretrained bias.
        use_bias : bool
            Add bias to the output tensor.
        name : str
            Name of this layer.
        """
        self.shape = (kw, kh, in_f, out_f)
        self.stride = stride
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.init_type = kernel_initializer

        name = str(name)

        if W is None:
            W = InitConvKernel.init_by_name(kw, kh, out_f, in_f, kernel_initializer)
        if b is None:
            b = np.zeros(out_f)



        self.name_conv = 'ConvKernel_{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, out_f) + name
        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)

        w_shape = self.W.shape.as_list()
        w_new_shape = [ np.prod(w_shape[:-1]), w_shape[-1] ]
        name_u = 'u_vec' + name
        self.u = tf.Variable(np.random.normal(size=[w_new_shape[0], 1]).astype(np.float32), dtype=tf.float32,
                             trainable=False,
                             shape=[w_new_shape[0], 1],
                             name=name_u)

        self.W_norm = spectral_normed_weight(self.W, name, u=self.u)
        params = [self.W, self.u]

        named_params_dict = {self.name_conv: self.W, name_u: self.u}
        regularize_params = [self.W]
        if use_bias:
            self.name_bias = 'ConvBias_{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, out_f) + name
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b
            if regularize_bias:
                regularize_params += [self.b]

        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def _forward(self, x):
        conv_out = tf.nn.conv2d(x, self.W_norm, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        if self.use_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.f is None:
            return conv_out
        return self.f(conv_out)

    def _training_forward(self, X):
        return self._forward(X)


    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]

        kw = params[ConvLayer.SHAPE][0]
        kh = params[ConvLayer.SHAPE][1]
        in_f = params[ConvLayer.SHAPE][2]
        out_f = params[ConvLayer.SHAPE][3]

        stride = params[ConvLayer.STRIDE]
        padding = params[ConvLayer.PADDING]
        activation = ActivationConverter.str_to_activation(params[ConvLayer.ACTIVATION])

        init_type = params[ConvLayer.INIT_TYPE]
        use_bias = params[ConvLayer.USE_BIAS]

        return ConvLayer(
            kw=kw, kh=kh, in_f=in_f, out_f=out_f,
            stride=stride, name=name, padding=padding, activation=activation,
            kernel_initializer=init_type, use_bias=use_bias
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: ConvLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self._name,
                ConvLayer.SHAPE: list(self.shape),
                ConvLayer.STRI: self.stride,
                ConvLayer.PADDING: self.padding,
                ConvLayer.ACTIVATION: ActivationConverter.activation_to_str(self.f),
                ConvLayer.USE_BIAS: self.use_bias,
                ConvLayer.INIT_TYPE: self.init_type
            }

        }

class UpConvLayerSpectral(SimpleForwardLayer):
    TYPE = 'UpConvLayer'
    SHAPE = 'shape'
    SIZE = 'size'
    PADDING = 'padding'
    ACTIVATION = 'activation'
    USE_BIAS = 'use_bias'
    INIT_TYPE = 'init_type'

    def __init__(self, kw, kh, in_f, out_f, name, size=(2, 2), padding='SAME', activation=tf.nn.relu,
                 kernel_initializer=InitConvKernel.HE, use_bias=True, regularize_bias=False, W=None, b=None):
        """
        Parameters
        ----------
        kw : int
            Kernel width.
        kh : int
            Kernel height.
        in_f : int
            Number of input feature maps. Treat as color channels if this layer
            is first one.
        out_f : int
            Number of output feature maps (number of filters).
        size : tuple
            Tuple of two ints - factors of the size of the output feature map.
            Example: feature map with spatial dimension (n, m) will produce
            output feature map of size (a*n, b*m) after performing up-convolution
            with `size` (a, b).
        padding : str
            Padding mode for convolution operation. Options: 'SAME', 'VALID' (case sensitive).
        activation : tensorflow function
            Activation function. Set None if you don't need activation.
        W : numpy array
            Filter's weights. This value is used for the filter initialization with pretrained filters.
        b : numpy array
            Bias' weights. This value is used for the bias initialization with pretrained bias.
        use_bias : bool
            Add bias to the output tensor.
        """
        # Shape is different from normal convolution since it's required by
        # transposed convolution. Output feature maps go before input ones.
        self.shape = (kw, kh, out_f, in_f)
        self.size = size
        self.strides = [1, *size, 1]
        self.padding = padding
        self.f = activation
        self.use_bias = use_bias
        self.init_type = kernel_initializer

        name = str(name)

        if W is None:
            W = InitConvKernel.init_by_name(kw, kh, in_f, out_f, kernel_initializer)
        if b is None:
            b = np.zeros(out_f)

        self.name_conv = 'UpConvKernel_{}x{}_out{}_in{}_id_'.format(kw, kh, out_f, in_f) + name
        self.W = tf.Variable(W.astype(np.float32), name=self.name_conv)

        w_shape = self.W.shape.as_list()
        w_new_shape = [ np.prod(w_shape[:-1]), w_shape[-1] ]
        name_u = 'u_vec' + name
        self.u = tf.Variable(np.random.normal(size=[w_new_shape[0], 1]).astype(np.float32), dtype=tf.float32,
                             trainable=False,
                             shape=[w_new_shape[0], 1],
                             name=name_u)

        self.W_norm = spectral_normed_weight(self.W, name, u=self.u)
        params = [self.W, self.u]

        named_params_dict = {self.name_conv: self.W, name_u: self.u}
        regularize_params = [self.W]

        if use_bias:
            self.name_bias = 'UpConvBias_{}x{}_in{}_out{}_id_'.format(kw, kh, in_f, out_f) + name
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b
            if regularize_bias:
                regularize_params += [self.b]

        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def _forward(self, x):
        out_shape = x.get_shape().as_list()
        out_shape[1] *= self.size[0]
        out_shape[2] *= self.size[1]
        # out_f
        out_shape[3] = self.shape[2]
        conv_out = tf.nn.conv2d_transpose(
            x, self.W_norm,
            output_shape=out_shape, strides=self.strides, padding=self.padding
        )
        if self.use_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)

        if self.f is None:
            return conv_out
        return self.f(conv_out)

    def _training_forward(self, X):
        return self._forward(X)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]

        kw = params[UpConvLayer.SHAPE][0]
        kh = params[UpConvLayer.SHAPE][1]
        in_f = params[UpConvLayer.SHAPE][3]
        out_f = params[UpConvLayer.SHAPE][2]

        padding = params[UpConvLayer.PADDING]
        size = params[UpConvLayer.SIZE]

        activation = ActivationConverter.str_to_activation(params[UpConvLayer.ACTIVATION])

        init_type = params[UpConvLayer.INIT_TYPE]
        use_bias = params[UpConvLayer.USE_BIAS]
        return UpConvLayer(
            kw=kw, kh=kh, in_f=in_f, out_f=out_f, size=size,
            name=name, padding=padding, activation=activation,
            kernel_initializer=init_type, use_bias=use_bias
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: UpConvLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self._name,
                UpConvLayer.SHAPE: list(self.shape),
                UpConvLayer.SIZE: self.size,
                UpConvLayer.PADDING: self.padding,
                UpConvLayer.ACTIVATION: ActivationConverter.activation_to_str(self.f),
                UpConvLayer.USE_BIAS: self.use_bias,
                UpConvLayer.INIT_TYPE: self.init_type
            }
        }


class DenseLayerSpectral(SimpleForwardLayer):
    TYPE = 'DenseLayer'
    INPUT_SHAPE = 'input_shape'
    OUTPUT_SHAPE = 'output_shape'
    ACTIVATION = 'activation'
    USE_BIAS = 'use_bias'
    INIT_TYPE = 'init_type'

    def __init__(self, in_d, out_d, name, activation=tf.nn.relu, mat_initializer=InitDenseMat.HE,
                 use_bias=True, regularize_bias=False, W=None, b=None):
        """
        Paremeters
        ----------
        in_d : int
            Dimensionality of the input vector. Example: 500.
        out_d : int
            Dimensionality of the output vector. Example: 100.
        activation : TensorFlow function
            Activation function. Set to None if you don't need activation.
        W : numpy ndarray
            Used for initialization the weight matrix.
        b : numpy ndarray
            Used for initialisation the bias vector.
        use_bias : bool
            Add bias to the output tensor.
        name : str
            Name of this layer.
        """
        self.input_shape = in_d
        self.output_shape = out_d
        self.f = activation
        self.use_bias = use_bias
        self.init_type = mat_initializer

        if W is None:
            W = InitDenseMat.init_by_name(in_d, out_d, mat_initializer)

        if b is None:
            b = np.zeros(out_d)

        name = str(name)
        self.name_dense = 'DenseMat_{}x{}_id_'.format(in_d, out_d) + name
        self.W = tf.Variable(W, name=self.name_dense)

        w_shape = self.W.shape.as_list()
        w_new_shape = [ np.prod(w_shape[:-1]), w_shape[-1] ]
        name_u = 'u_vec' + name
        self.u = tf.Variable(np.random.normal(size=[w_new_shape[0], 1]).astype(np.float32), dtype=tf.float32,
                             trainable=False,
                             shape=[w_new_shape[0], 1],
                             name=name_u)

        self.W_norm = spectral_normed_weight(self.W, name, u=self.u)
        params = [self.W, self.u]


        named_params_dict = {self.name_dense: self.W, name_u: self.u}
        regularize_params = [self.W]

        if use_bias:
            self.name_bias = 'DenseBias_{}x{}_id_'.format(in_d, out_d) + name
            self.b = tf.Variable(b.astype(np.float32), name=self.name_bias)
            params += [self.b]
            named_params_dict[self.name_bias] = self.b
            if regularize_bias:
                regularize_params += [self.b]

        super().__init__(name, params=params,
                         regularize_params=regularize_params,
                         named_params_dict=named_params_dict
        )

    def _forward(self, x):
        out = tf.matmul(x, self.W_norm)
        if self.use_bias:
            out = out + self.b
        if self.f is None:
            return out
        return self.f(out)

    def _training_forward(self, X):
        return self._forward(X)

    @staticmethod
    def build(params: dict):
        name = params[MakiRestorable.NAME]
        input_shape = params[DenseLayer.INPUT_SHAPE]
        output_shape = params[DenseLayer.OUTPUT_SHAPE]

        activation = ActivationConverter.str_to_activation(params[DenseLayer.ACTIVATION])

        init_type = params[DenseLayer.INIT_TYPE]
        use_bias = params[DenseLayer.USE_BIAS]

        return DenseLayer(
            in_d=input_shape, out_d=output_shape,
            activation=activation, name=name,
            mat_initializer=init_type, use_bias=use_bias
        )

    def to_dict(self):
        return {
            MakiRestorable.FIELD_TYPE: DenseLayer.TYPE,
            MakiRestorable.PARAMS: {
                MakiRestorable.NAME: self._name,
                DenseLayer.INPUT_SHAPE: self.input_shape,
                DenseLayer.OUTPUT_SHAPE: self.output_shape,
                DenseLayer.ACTIVATION: ActivationConverter.activation_to_str(self.f),
                DenseLayer.USE_BIAS: self.use_bias,
                DenseLayer.INIT_TYPE: self.init_type
            }
        }