import time
import os

import tensorflow as tf
import numpy as np
from skimage.io import imread
from scipy.ndimage import rotate

class unet3d():

    def __init__(self, params):
        self.width = params['width']
        self.height = params['height']
        self.depth = params['depth']
        self.nchannels = params['nchannels']
        self.channels = params['channels']
        self.nepochs = params['nepochs']
        self.batchsize = params['batchsize']
        self.learning_rate = params['learning_rate']
        self.restore = params['restore']
        self.latent_size = params['latent_size']
        self.enc_sizes = params['enc_sizes']
        self.dec_sizes = params['dec_sizes']
        self.droprate = params['droprate']
        self.stdev = params['stdev']
        self.used = dict()

    def leaky_relu(self, x, alpha=0.2):

        return tf.maximum(x*alpha, x)

    def dropout(self, x, rate, is_train):
        if is_train:
            x = tf.nn.dropout(x, rate)
        return x

    def get_init(self, stdev):
        return tf.truncated_normal_initializer(stddev=stdev)

    
    def create_encoder(self, images, is_train):
        '''The encoder part of the network, must include weight tensors
        for concatenation, get the network sizes from the class'''
    
        layers = list()
        ph = images
        layers.append(images)
        for i, ei in enumerate(self.enc_sizes):
            nfilters = ei[0]
            ksize = ei[1]
            strides = 2
            padding = 'same'
            print(ei)
            name = 'encoder-layer-{}-0'.format(nfilters)
            h = tf.layers.conv3d(ph, nfilters, ksize, strides=strides,
                                 padding=padding,
                                 kernel_initializer=self.get_init(self.stdev),
                                 name=name, activation=None)
            h = self.leaky_relu(h)
            if self.droprate > 0:
                h = self.dropout(h, self.droprate, is_train)

            name = 'encoder-layer-{}'.format(nfilters)
            # h = tf.layers.conv3d(h, nfilters, ksize, strides=1,
            #                      padding=padding,
            #                      kernel_initializer=self.get_init(self.stdev),
            #                      name=name, activation=None)
            
            # h = self.leaky_relu(h)
            # if self.droprate > 0:
            #     h = self.dropout(h, self.droprate, is_train)
    
            layers.append(h) ## only append the second convolution
            ### use identity to rename the final tensor 
            h = tf.identity(h, name='encoder-{}'.format(nfilters))
            ph = h
        ### end the for loop for encoder layers
    
        self.encoder_layers = layers
        self.encoder = h
        

    def create_decoder(self):
        '''decoder part of the network, need weights to concatenate'''
        ### transpose, concat, conv, conv
        layers = list()
        ph = self.encoder
        for i, di in enumerate(self.dec_sizes):
            nfilters = di[0]
            ksize = di[1]
            strides = 2
            name = 'decoder-layer-{}'.format(nfilters)
            h = tf.layers.conv3d_transpose(ph, nfilters, ksize, strides,
                                           padding = 'same',
                                           activation=None,
                                           name=name)

            h = self.leaky_relu(h)
            nl = len(self.encoder_layers) - i - 2
            print(nl, self.encoder_layers[nl])
            h = tf.concat([h, self.encoder_layers[nl]], -1,
                          name='concat-{}'.format(nfilters))

            print('after concat', h)
            h = tf.layers.conv3d(h, nfilters, ksize, strides=1, padding='same',
                          kernel_initializer=self.get_init(self.stdev),
                          name='decoder-conv-{}-1'.format(nfilters),
                          activation=None)

            h = self.leaky_relu(h)
            print(h)
            # h = tf.layers.conv3d(h, nfilters, ksize, strides=1, padding='same',
            #               kernel_initializer=self.get_init(self.stdev),
            #               name='decoder-conv-{}-2'.format(nfilters),
            #               activation=None)

            # if i < (len(self.dec_sizes) - 2):
            #     h = self.leaky_relu(h)
            # print(h)
            ph = h    
            h = tf.identity(h, name='decoder-{}'.format(nfilters))

        self.decoder_sigmoid = tf.sigmoid(h, name='decoder-sigmoid')
        self.decoder = h
            
    def create_loss(self, batch_mask):
        ''' calculate the loss of the network - cross entropy of input with output pixels'''
        loss =  tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_mask,
                                                        logits=self.decoder,
                                                        name='sce_loss')
        
        s = tf.abs(tf.reduce_sum(self.decoder_sigmoid))
        td = tf.reduce_mean(tf.reduce_sum(tf.square(self.decoder_sigmoid - batch_mask), axis=(1,2,3,4)))
        print("mmse loss", td.shape)
        self.loss = tf.reduce_mean(tf.reduce_sum(loss, axis=(1,2,3,4))) + 0*td + 0*s


    def create_opt(self):
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                     name='adam_opt').minimize(self.loss)

        self.opt= opt


    def normalize_images(self, images):

        ymax = images.max(axis=(1,2,3), keepdims=True)
        ymin = images.min(axis=(1,2,3), keepdims=True)
    
        y = (images - ymin)/(ymax - ymin + 0.00001)
        return y

    def reorder_images(self, images):
        y = np.moveaxis(images, 1, -1)
        y = np.moveaxis(y, 1, -1)

        return y

    def read_images(self, imagepath):
        x0 = imread(imagepath)
        x1 = self.reorder_images(x0)
        x = self.normalize_images(x1)
        x = x[:,0:256, 0:256, :, :]
        self.x = x

    def read_mm(self, mmpath, shape):
        self.x = np.memmap(mmpath, dtype=np.float32, mode='r', shape=shape)
        
    def augment_batch(self, b, nr):
        n = b.shape[0]
        ab = 0.0*b
        nused = 0
        for i, r in enumerate(nr):
            s = b[i]  # this loses the first dimension
            angle = 9*np.random.randint(0,360)
            #flips = np.random.randint(0,2,3)
            j = (r, angle)
            if j in self.used:
                ab[i] = self.used[j]
                nused += 1
            else:
                s = rotate(s, angle, (0,1), reshape=False, order=1)
                #print(s.shape)
#                 flips = np.random.randint(0,2,3)
#                 for k, f in enumerate(flips):
#                     if f:
#                         s = np.flip(s, k) 
#                 ab[i] = s
                self.used[j] = s
            #print(angle, flips)
        #print("Length of used:", len(self.used),  nused)
        return ab 
    
    def get_batch(self, n, augment=True, training=True):
        if training:
            s = self.xtrain
        else:
            s = self.x
        nx = s.shape[0]
        nr = np.random.randint(0, nx, n)
        b = s[nr,:,:,:, :]

        if augment:
            self.augment_batch(b , nr)

        return b
        
    def set_validation(self, n):
        self.xvalid = self.x[-n:]
        self.xtrain = self.x[:-n]
    
    def train(self, niterations):
        for i in range(10):
            nr = np.random.randint(0,36, 1)
            bx = np.expand_dims(u.x[nr,:,:,:,0], -1)
            bm = np.expand_dims(u.x[nr,:,:,:,1], -1)
            print(bx.shape)
            _, res = sess.run([u.opt, u.decoder_sigmoid], feed_dict={images:bx, masks:bm})
            plt.imshow(res)
    
        
        
        
        


if __name__ == '__main__':

    pass
    #params = {'width
    #u = unet3d(params)


    