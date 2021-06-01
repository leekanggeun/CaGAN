import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np

class MIOU(tf.keras.metrics.Metric):
    def __init__(self, name='Mean_IOU', **kwargs):
        super(MIOU, self).__init__(name=name, **kwargs)
        self.mean_iou = self.add_weight(name='miou', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        tp = tf.reduce_sum(tf.cast(tp, self.dtype))
        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        fp = tf.reduce_sum(tf.cast(fp, self.dtype))
        fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        fn = tf.reduce_sum(tf.cast(fn, self.dtype))
        dice_score = tp/(tp+fp+fn)
        self.mean_iou.assign_add(dice_score)
    def result(self):
        return self.mean_iou
    def reset_states(self):
        K.set_value(self.mean_iou, 0.0)

class PSNR(tf.keras.metrics.Metric):
    def __init__(self, name='psnr', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='value', initializer='zeros')
    def update_state(self, y_true, y_pred):
        self.psnr.assign_add(tf.image.psnr(y_true, y_pred, max_val=255.0))
    def result(self):
        return self.psnr
    def reset_state(self):
        K.set_value(self.psnr, 0.0)

class SSIM(tf.keras.metrics.Metric):
    def __init__(self, name='ssim', **kwargs):
        super(SSIM, self).__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name='value', initializer='zeros')
    def update_state(self, y_true, y_pred):
        self.ssim.assign_add(tf.image.ssim(y_true, y_pred, max_val=255.0))
    def result(self):
        return self.ssim
    def reset_state(self):
        K.set_value(self.ssim, 0.0)


class Custom_SSIM_Loss(object):
    def __init__(self, k1=0.01, k2=0.02, L=1, window_size=11):
        self.k1 = k1
        self.k2 = k2           # constants for stable
        self.L = L             # the value range of input image pixels
        self.WS = window_size
        self.window = self._tf_fspecial_gauss(size=self.WS)  # output size is (window_size, window_size, 1, 1)
    def _tf_fspecial_gauss(self, size, sigma=1.5):
        """Function to mimic the 'fspecial' gaussian MATLAB function"""
        x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g / tf.reduce_sum(g)

    def ssim_loss(self, img1, img2):
        """
        The function is to calculate the ssim score
        """
        #import pdb
        #pdb.set_trace()

        (_, _, _, channel) = img1.shape.as_list()

        window = tf.tile(self.window, [1, 1, channel, 1])

        # here we use tf.nn.depthwise_conv2d to imitate the group operation in torch.nn.conv2d 
        mu1 = tf.nn.depthwise_conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
        mu2 = tf.nn.depthwise_conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        img1_2 = img1*img1#tf.pad(img1*img1, [[0,0], [0, self.WS//2], [0, self.WS//2], [0,0]], "CONSTANT")
        sigma1_sq = tf.subtract(tf.nn.depthwise_conv2d(img1_2, window, strides = [1 ,1, 1, 1], padding = 'VALID') , mu1_sq)
        img2_2 = img2*img2#tf.pad(img2*img2, [[0,0], [0, self.WS//2], [0, self.WS//2], [0,0]], "CONSTANT")
        sigma2_sq = tf.subtract(tf.nn.depthwise_conv2d(img2_2, window, strides = [1, 1, 1, 1], padding = 'VALID') ,mu2_sq)
        img12_2 = img1*img2#tf.pad(img1*img2, [[0,0], [0, self.WS//2], [0, self.WS//2], [0,0]], "CONSTANT")
        sigma1_2 = tf.subtract(tf.nn.depthwise_conv2d(img12_2, window, strides = [1, 1, 1, 1], padding = 'VALID') , mu1_mu2)

        c1 = (self.k1*self.L)**2
        c2 = (self.k2*self.L)**2

        ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))

        return tf.reduce_mean(ssim_map)