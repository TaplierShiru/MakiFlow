import tensorflow as tf
import numpy as np
from makiflow.layers import MakiLayer,ConvLayer, ActivationLayer, BatchNormLayer
from makiflow.save_recover.activation_converter import ActivationConverter

# Reference: https://arxiv.org/pdf/1602.07261.pdf page 8, Figure 16

class InceptionA(MakiLayer):

	def __init__(self,in_f,out_f,activation=tf.nn.relu,name='inception_a',alpha=0.1):
		"""
		Parameters
		----------
		in_f : int
			Number of the input feature maps relative to the first convolution in the block.
		out_f : list of int
			Numbers of the outputs feature maps of the convolutions in the block.
			out_f = [
				Conv1,
				Conv2,
				Conv3,
			]
		
		Notes
		-----
			Data flow scheme: 
			(left)	     /->Conv1(1x1)-----------------|
			(mid)input->|->Conv1(1x1)->Conv1-----------|+(concate)->Conv(1x1)=last_conv_output
			(right)	     \-> Conv1(1x1)->Conv2->Conv3--|
			Where two branches are summed together:
			final_output = input + last_conv_output.
		"""
		assert(len(out_f) == 3)
		MakiLayer.__init__(self)
		self.name = name
		self.in_f = in_f
		self.out_f = out_f
		self.f = activation
		self.alpha = alpha

		self.layers = []
		# Left branch
		self.conv1_L_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name=name+'_conv1_L_1')
		self.batch_norm1_L_1 = BatchNormLayer(D=out_f[0],name=name+'_batch_norm1_L_1')
		self.activ1_L_1 = ActivationLayer(activation=activation)
		self.layers += [self.conv1_L_1,self.batch_norm1_L_1,self.activ1_L_1]

		# Mid branch
		self.conv1_M_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name=name+'_conv1_M_1')
		self.batch_norm1_M_1 = BatchNormLayer(D=out_f[0],name=name+'_batch_norm1_M_1')
		self.activ1_M_1 = ActivationLayer(activation=activation)
		self.layers += [self.conv1_M_1,self.batch_norm1_M_1,self.activ1_M_1]

		self.conv1_M_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[0],activation=None,name=name+'_conv1_M_2')
		self.batch_norm1_M_2 = BatchNormLayer(D=out_f[0],name=name+'_batch_norm1_M_2')
		self.activ1_M_2 = ActivationLayer(activation=activation)
		self.layers += [self.conv1_M_2,self.batch_norm1_M_2,self.activ1_M_2]

		# Right branch
		self.conv1_R_1 =  ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name=name+'_conv1_R_1')
		self.batch_norm1_R_1 = BatchNormLayer(D=out_f[0],name=name+'_batch_norm1_R_1')
		self.activ1_R_1 = ActivationLayer(activation=activation)
		self.layers += [self.conv1_R_1,self.batch_norm1_R_1,self.activ1_R_1]

		self.conv1_R_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[1],activation=None,name=name+'_conv1_R_2')
		self.batch_norm1_R_2 = BatchNormLayer(D=out_f[1],name=name+'_batch_norm1_R_2')
		self.activ1_R_2 = ActivationLayer(activation=activation)
		self.layers += [self.conv1_R_2,self.batch_norm1_R_2,self.activ1_R_2]

		self.conv1_R_3 = ConvLayer(kw=3,kh=3,in_f=out_f[1],out_f=out_f[2],activation=None,name=name+'_conv1_R_3')
		self.batch_norm1_R_3 = BatchNormLayer(D=out_f[2],name=name+'_batch_norm1_R_3')
		self.activ1_R_3 = ActivationLayer(activation=activation)
		self.layers += [self.conv1_R_3,self.batch_norm1_R_3,self.activ1_R_3]

		# Connect three branch
		self.conv2_connect = ConvLayer(kw=1,kh=1,in_f=out_f[0]*2+out_f[2],out_f=in_f,activation=None,name=name+'_conv2_connect')
		self.layers += [self.conv2_connect]

		self.named_params_dict = {}
		
		for layer in self.layers:
			self.named_params_dict.update(layer.get_params_dict())
	
	def forward(self,X,is_training=False):
		FX = X

		# Left branch
		LX = self.conv1_L_1.forward(FX,is_training)
		LX = self.batch_norm1_L_1.forward(LX,is_training)
		LX = self.activ1_L_1.forward(LX,is_training)
			
		# Mid branch
		MX = self.conv1_M_1.forward(FX,is_training)
		MX = self.batch_norm1_M_1.forward(MX,is_training)
		MX = self.activ1_M_1.forward(MX,is_training)

		MX = self.conv1_M_2.forward(MX,is_training)
		MX = self.batch_norm1_M_2.forward(MX,is_training)
		MX = self.activ1_M_2.forward(MX,is_training)

		# Right branch
		RX = self.conv1_R_1.forward(FX,is_training)
		RX = self.batch_norm1_R_1.forward(RX,is_training)
		RX = self.activ1_R_1.forward(RX,is_training)

		RX = self.conv1_R_2.forward(RX,is_training)
		RX = self.batch_norm1_R_2.forward(RX,is_training)
		RX = self.activ1_R_2.forward(RX,is_training)

		RX = self.conv1_R_3.forward(RX,is_training)
		RX = self.batch_norm1_R_3.forward(RX,is_training)
		RX = self.activ1_R_3.forward(RX,is_training)

		# Connect branches

		FX = tf.concat([LX,MX,RX],axis=3)

		FX = self.conv2_connect.forward(FX,is_training)

		FX = FX * self.alpha
		
		return FX + X

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params
	
		
	def get_params_dict(self):
		return self.named_params_dict


	def to_dict(self):
		return {
			'type':'ResnetInceptionBlockA',
			'params':{
				'name': self.name,
				'in_f': self.in_f,
				'out_f': self.out_f,
				'activation': ActivationConverter.activation_to_str(self.f),
				'alpha' : self.alpha,
			}
		}



		