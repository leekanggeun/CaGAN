import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow_addons as tfa
import sys
from network import Generator, Discriminator
from ..utils.metrics import PSNR, SSIM, Custom_SSIM_Loss
from ..utils.scheduler import CustomSchedule

class Trainer(Model):
    def __init__(self, args):
        super(Trainer, self).__init__()
        # Initialization
        self.initializer = 'truncated_normal'
        
        # Models
        self.G1 = Generator(initializer=self.initializer) # noisy to clean
        self.G2 = Generator(initializer=self.initializer) # clean to noisy
        self.D1 = Discriminator(initializer=self.initializer) # noisy dis
        self.D2 = Discriminator(initializer=self.initializer) # clean dis

        # Scheduler 
        self.G_scheduler = CustomSchedule(1e-4, args.iter, 0.995) # 0.5% decay
        self.D_scheduler = CustomSchedule(1e-4, args.iter, 0.995) # 0.5% decay

        # Optimizers
        self.G_optimizer = tf.keras.optimizers.Adam(learning_rate=self.G_scheduler, beta_1=0.5, beta_2=0.9)
        self.D_optimizer = tf.keras.optimizers.Adam(learning_rate=self.D_scheduler, beta_1=0.5, beta_2=0.9)
        
        # Trackers
        self.gloss_tracker = [tf.keras.metrics.Mean(name="Gloss"), tf.keras.metrics.Mean(name="Cycle"), tf.keras.metrics.Mean(name="idt"), tf.keras.metrics.Mean(name="Str"), tf.keras.metrics.Mean(name="SSIM")]
        self.dloss_tracker = [tf.keras.metrics.Mean(name="Dloss")]
        self.val_tracker = [PSNR(), SSIM()]

        # SSIM loss
        self.ssim = Custom_SSIM_Loss()

        # Reset tracker
        for tracker in self.gloss_tracker:
            tracker.reset_state()
        for tracker in self.dloss_tracker:
            tracker.reset_state()
        for tracker in self.val_tracker:
            tracker.reset_state()

    def compile(self, **kwargs):
        self._configure_steps_per_execution(1)
        self._reset_compile_cache()
        self._is_compiled =True
        self.loss = {}

    def call(self, noisy, training=True):
        y_hat = self.G1(noisy, training=training)
        return y_hat

    def train_generator(self, clean, noisy):
        with tf.GradientTape(persistent=True) as tape:
            y_hat = self.G1(noisy, training=True) # clean
            x_bar = self.G2(y_hat, training=True) # noisy
            x_hat = self.G2(clean, training=True) # noisy
            y_bar = self.G1(x_hat, training=True) # clean
            
            noisy_gloss = tf.reduce_mean(tf.square(self.D1(x_hat)-1))
            clean_gloss = tf.reduce_mean(tf.square(self.D2(y_hat)-1))
            gloss = noisy_gloss+clean_gloss
            
            # Cycle loss
            cycle_loss = tf.reduce_mean(tf.abs(noisy-x_bar))+tf.reduce_mean(tf.abs(clean-y_bar))
            
            # Identity loss
            idt_loss = tf.reduce_mean(tf.abs(self.G2(noisy, training=True)-noisy))+tf.reduce_mean(tf.abs(self.G1(clean, training=True)-clean))

            # Structural restoration loss
            L1_loss = tf.reduce_mean(tf.abs(self.G1(noisy, training=True)-clean))
            ssim_loss = 1-self.ssim.ssim_loss(clean, y_hat)
            str_loss = 0.84*L1_loss + 0.16*ssim_loss
            # Total loss
            loss = gloss + 0.1*cycle_loss + 0.1*idt_loss + 10*str_loss
        gradient_g = tape.gradient(loss, self.G1.trainable_variables+self.G2.trainable_variables)
        self.G_optimizer.apply_gradients(zip(gradient_g, self.G1.trainable_variables+self.G2.trainable_variables))
        self.gloss_tracker[0].update_state(gloss)
        self.gloss_tracker[1].update_state(cycle_loss)
        self.gloss_tracker[2].update_state(idt_loss)
        self.gloss_tracker[3].update_state(str_loss)
        return {"Gloss":self.gloss_tracker[0].result(), 
                "Cycle":self.gloss_tracker[1].result(), 
                "Idt":self.gloss_tracker[2].result(), 
                "Str":self.gloss_tracker[3].result()}

    def train_discriminator(self, clean, noisy):
        with tf.GradientTape(persistent=True) as tape:
            # Discriminator loss
            fake_noisy = self.G2(clean, training=False)
            fake_clean = self.G1(noisy, training=False)
            noisy_dloss = tf.reduce_mean(tf.square(self.D1(noisy, training=True)-1))+tf.reduce_mean(tf.square(self.D1(fake_noisy, training=True)))
            clean_dloss = tf.reduce_mean(tf.square(self.D2(clean, training=True)-1))+tf.reduce_mean(tf.square(self.D2(fake_clean, training=True)))
            
            # Total loss
            loss = noisy_dloss+clean_dloss
        gradient_d = tape.gradient(loss, self.D1.trainable_variables+self.D2.trainable_variables)
        self.D_optimizer.apply_gradients(zip(gradient_d, self.D1.trainable_variables+self.D2.trainable_variables))
        self.dloss_tracker[0].update_state(loss)
        return {"Dloss":self.dloss_tracker[0].result()}
    

    def train_step(self, data):
        clean, noisy = data
        dloss = self.train_discriminator(clean, noisy)
        gloss = self.train_generator(clean, noisy)
        return {**gloss, **dloss}
    
    def predict(self, noisy):
        y_hat = tf.clip_by_value(self.G1(noisy, training=False), 0.0, 1.0)*255
        return y_hat

    def test_step(self, data):
        # Label range [0,255], prediction range [-1, 1]
        data, label = data # label 1, 512, 512
        predict = tf.squeeze(self.predict(data),0) 
        label = tf.expand_dims(tf.squeeze(label,0),-1)
        self.val_tracker[0].update_state(label, predict)
        self.val_tracker[1].update_state(label, predict)
        return {"PSNR":self.val_tracker[0].result()/32, 
                "SSIM":self.val_tracker[1].result()/32}

        
