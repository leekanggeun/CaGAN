import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class CA(layers.Layer): 
    def __init__(self, num_filters, initializer='he_normal'):
        super(CA, self).__init__()
        self.G = layers.GlobalAveragePooling2D()
        self.num_filters = num_filters
        self.scale = 4
        self.WD = layers.Conv2D(int(num_filters/self.scale), (1,1), (1,1), activation=None, padding="same", use_bias=True, kernel_initializer=initializer)
        self.WU = layers.Conv2D(num_filters, (1,1), (1,1), activation=None, padding="same", use_bias=True, kernel_initializer=initializer)

    def call(self, x):
        h = self.G(x)
        h = h[:, tf.newaxis, tf.newaxis, :]
        h = tf.nn.relu(self.WD(h))
        h = tf.nn.sigmoid(self.WU(h))
        h = x*h
        return h

# CrissCrossAttention
class CCSA(layers.Layer): 
    def __init__(self, num_filters, batch_size, initializer='he_normal'):
        super(CCSA, self).__init__()
        self.im_ch = int(num_filters//8)
        self.f = layers.Conv2D(self.im_ch, (1,1), (1,1), activation=None, padding="same", use_bias=True, kernel_initializer=initializer)
        self.g = layers.Conv2D(self.im_ch, (1,1), (1,1), activation=None, padding="same", use_bias=True, kernel_initializer=initializer)
        self.h = layers.Conv2D(num_filters, (1,1), (1,1), activation=None, padding="same", use_bias=True, kernel_initializer=initializer)
        self.INF = INF
        self.gamma = tf.Variable(tf.constant(0.0), trainable=True)

    def call(self, x):
        _, h, w, c = x.shape # 
        fx = self.f(x)
        gx = self.g(x)
        hx = self.h(x)
        #proj_fx_H = tf.transpose(tf.reshape(tf.transpose(fx, perm=[0, 2, 3, 1]), [-1, c, h]), perm=[0, 2, 1]) # batch_size*width, height, channel
        #proj_fx_W = tf.transpose(tf.reshape(tf.transpose(fx, perm=[0, 1, 3, 2]), [-1, c, w]), perm=[0, 2, 1]) # batch_size*height, width, channel
        proj_fx_H = tf.reshape(tf.transpose(fx, perm=[0, 2, 1, 3]), [-1, h, self.im_ch]) # batch_size*width, height, channel
        proj_fx_W = tf.reshape(fx, [-1, w, self.im_ch]) # batch_size*height, width, channel
        proj_gx_H = tf.reshape(tf.transpose(gx, perm=[0, 2, 3, 1]), [-1, self.im_ch, h]) # batch_size*width, channel, height
        proj_gx_W = tf.reshape(tf.transpose(gx, perm=[0, 1, 3, 2]), [-1, self.im_ch, w]) # batch_size*height, channel, width
        proj_hx_H = tf.reshape(tf.transpose(hx, perm=[0, 2, 3, 1]), [-1, c, h]) # batch_size*width, channel, height
        proj_hx_W = tf.reshape(tf.transpose(hx, perm=[0, 1, 3, 2]), [-1, c, w]) # batch_size*height, channel, width

        energy_h = tf.transpose(tf.reshape(tf.matmul(proj_fx_H, proj_gx_H)+self.INF(h), [-1, w, h, h]), perm=[0,2,1,3]) # batch_size, height, width, height
        energy_w = tf.matmul(proj_fx_W, proj_gx_W)
        energy_w = tf.reshape(energy_w, [-1, h, w, w]) # batch_size, height, width, width
        #energy_w = tf.reshape(tf.matmul(proj_fx_W, proj_gx_W), [-1, h, w, w]) # batch_size, height, width, width
        energy = tf.nn.softmax(tf.concat([energy_h,energy_w], -1), axis=3)
        att_h, att_w = tf.split(energy, num_or_size_splits=[h, w], axis=3)
        att_h = tf.reshape(tf.transpose(att_h, perm=[0, 2, 1, 3]), [-1, h, h]) # batch_size*width, height, height
        att_w = tf.reshape(att_w, [-1, w, w]) # batch_size*height, width, width
        out_h = tf.transpose(tf.reshape(tf.linalg.matmul(proj_hx_H, tf.transpose(att_h, perm=[0, 2, 1])), [-1, w, c, h]), perm=[0,3,1,2]) # batch_size, height, width, channel
        out_w = tf.transpose(tf.reshape(tf.linalg.matmul(proj_hx_W, tf.transpose(att_w, perm=[0, 2, 1])), [-1, h, c, w]), perm=[0,1,3,2]) # batch_size, height, width, channel
        return self.gamma*(out_h+out_w)+x
        



def INF(H):
    return tf.expand_dims(tf.eye(H)*tf.constant(3.4e+38),0)
        
        