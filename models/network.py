import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras import Model
import sys
sys.path.append('/home/Alexandrite/leekanggeun/CVPR/CaGAN')
from utils.attention_module import CA, CCSA

class FEN(Model):
    def __init__(self, num_filters, initializer='he_normal'):
        super(FEN, self).__init__()
        self.n_layers = 12
        self.h = []
        self.activation = layers.LeakyReLU(0.2)
        for i in range(0,self.n_layers):
            self.h.append(layers.Conv2D(num_filters, (3,3), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=initializer))
        self.CA_layer = CA(64*12, initializer)
        
    def call(self, inputs):
        y = self.h[0](inputs)
        skip = y
        for h in self.h[1:]:
            y = h(y)
            skip = tf.concat([skip, y], axis=-1)
        out = self.CA_layer(skip)
        return out

class Generator(Model):
    def __init__(self, num_filters=64, initializer='he_normal'):
        super(Generator, self).__init__()
        self.activation = layers.LeakyReLU(0.2)
        self.h = FEN(num_filters, initializer)
        self.h1 = layers.Conv2D(num_filters, (1,1), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=initializer)
        self.h2_1 = layers.Conv2D(num_filters, (1,1), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=initializer)
        self.h2_2 = layers.Conv2D(num_filters, (3,3), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=initializer)
        self.h3_1 = layers.Conv2D(num_filters, (1,1), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=initializer)
        self.h3_2 = layers.Conv2D(num_filters, (5,5), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=initializer)
        self.CCSA1 = CCSA(num_filters, initializer)
        self.CCSA2 = CCSA(num_filters, initializer)
        self.CCSA3 = CCSA(num_filters, initializer)
        self.h4 = layers.Conv2D(num_filters, (3,3), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=initializer)
        self.h5 = layers.Conv2D(num_filters, (3,3), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=initializer)
        self.h6 = layers.Conv2D(1, (3,3), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=initializer)

    def call(self, input):
        h = self.h(input)
        h1 = self.CCSA1(self.h1(h))
        h2 = self.CCSA2(self.h2_2(self.h2_1(h)))
        h3 = self.CCSA3(self.h3_2(self.h3_1(h)))
        h = tf.concat([h1, h2, h3], axis=-1)
        h = self.h4(h)
        h = self.h5(h)
        h = self.h6(h)
        h = h+tfa.image.mean_filter2d(input, filter_shape=[3,3])
        return h

class Discriminator(Model):
    def __init__(self, start_neuron=64, initializer='he_normal'):
        super(Discriminator, self).__init__()
        self.start_neuron = start_neuron
        self.initializer = initializer
        self.activation = layers.LeakyReLU(0.2)
        self.conv1 = layers.Conv2D(self.start_neuron, (3,3), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=self.initializer) # 64
        self.conv2 = layers.Conv2D(self.start_neuron, (3,3), (2,2), activation=self.activation, padding="same", use_bias=True, kernel_initializer=self.initializer) # 32
        self.conv3 = layers.Conv2D(self.start_neuron*2, (3,3), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=self.initializer) # 16
        self.conv4 = layers.Conv2D(self.start_neuron*2, (3,3), (2,2), activation=self.activation, padding="same", use_bias=True, kernel_initializer=self.initializer) # 8
        self.conv5 = layers.Conv2D(self.start_neuron*4, (3,3), (1,1), activation=self.activation, padding="same", use_bias=True, kernel_initializer=self.initializer) # 4
        self.conv6 = layers.Conv2D(self.start_neuron*4, (3,3), (2,2), activation=self.activation, padding="same", use_bias=True, kernel_initializer=self.initializer) # 2
        self.CCSA1 = CCSA(self.start_neuron*4, initializer)
        self.fcn1 = layers.Dense(1024, activation=self.activation)
        self.fcn2 = layers.Dense(1)
    def call(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.CCSA1(h)
        h = layers.Flatten()(h)
        h = self.fcn1(h)
        h = self.fcn2(h)
        return h
