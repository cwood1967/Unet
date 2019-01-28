import time
import os

import tensorflow as tf
import numpy as np
from skimage.io import imread
from scipy.ndimage import rotate
from skimage.morphology import binary_erosion
from skimage.morphology import binary_dilation
from skimage.morphology import erosion
from skimage.morphology import dilation

class unet2d():

    def __init__(self, params):
        self.width = params['width']
        self.height = params['height']
        self.nchannels = params['nchannels']
        self.channels = params['channels']
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
        #return tf.truncated_normal_initializer(stddev=stdev)
        return None
    
    def create_encoder(self, images, is_train):
        '''The encoder part of the network, must include weight tensors
        for concatenation, get the network sizes from the class'''
    
        layers = list()
        ph = images
        layers.append(images)
        for i, ei in enumerate(self.enc_sizes):
            nfilters = ei[0]
            ksize = ei[1]
            strides = ei[2]
            padding = 'same'
            print(ei, type(strides))
            name = 'encoder-layer-{}-0'.format(nfilters)
            ph = layers[-1]
            h = tf.layers.conv2d(ph, nfilters, ksize, strides=strides,
                                 padding=padding, dilation_rate=1,
                                 kernel_initializer=self.get_init(self.stdev),
                                 name=name, activation=None)

            #h = self.leaky_relu(h)
            if i < (len(self.enc_sizes) - 1):
                h = tf.nn.relu(h)
            if self.droprate > 0:
                h = self.dropout(h, self.droprate, is_train)

            name = 'encoder-layer-{}'.format(nfilters)
            h = tf.layers.conv2d(h, nfilters, ksize, strides=1, dilation_rate=2,
                                 padding=padding,
                                 kernel_initializer=self.get_init(self.stdev),
                                 name=name, activation=None)

            
            if i < (len(self.enc_sizes) - 1):
                h = tf.nn.relu(h)
            # if self.droprate > 0:
            #     h = self.dropout(h, self.droprate, is_train)
    
            layers.append(h) ## only append the second convolution
            ### use identity to rename the final tensor 

        ### end the for loop for encoder layers
        h = tf.identity(h, name='encoder-{}'.format(nfilters))    
        self.encoder_layers = layers
        self.encoder = h
        

    def create_decoder(self):
        '''decoder part of the network, need weights to concatenate'''
        ### transpose, concat, conv, conv
        layers = list()
        ph = self.encoder
        layers.append(self.encoder)
        for i, di in enumerate(self.dec_sizes):
            nfilters = di[0]
            ksize = di[1]
            strides = di[2]
            name = 'decoder-layer-{}'.format(nfilters)
            ph = layers[-1]
            h = tf.layers.conv2d_transpose(ph, nfilters, ksize, strides,
                                           padding = 'same',
                                           activation=None,
                                           name=name)

            #h = self.leaky_relu(h)
            #if i < (len(self.dec_sizes) - 1):
            #    h = tf.nn.relu(h)
            nl = len(self.encoder_layers) - i - 2
            print(h)
            print(nl, self.encoder_layers[nl])
            h = tf.concat([h, self.encoder_layers[nl]], -1,
                          name='concat-{}'.format(nfilters))

            print('after concat', h)
            h = tf.layers.conv2d(h, nfilters, ksize, strides=1, padding='same',
                                 dilation_rate=2,
                          kernel_initializer=self.get_init(self.stdev),
                          name='decoder-conv-{}-1'.format(nfilters),
                          activation=None)
            if i < (len(self.dec_sizes) - 2):
                #h = self.leaky_relu(h)
                h = tf.nn.relu(h)
            print(h)

            h = tf.layers.conv2d(h, nfilters, ksize, strides=1, padding='same',
                          kernel_initializer=self.get_init(self.stdev),
                          name='decoder-conv-{}-2'.format(nfilters),
                          activation=None)

            if i < (len(self.dec_sizes) - 2):
                h = self.leaky_relu(h)

            layers.append(h)
                # print(h)
            #ph = h
            
        h = tf.identity(h, name='decoder-{}'.format(nfilters))
        self.decoder_sigmoid = tf.sigmoid(h, name='decoder-sigmoid')
        self.decoder_softmax = tf.nn.softmax(h, dim=-1, name='decoder-softmax')
        self.decoder = h
            
    def create_loss(self, batch_mask):
        ''' calculate the loss of the network - cross entropy of input with output pixels'''
        loss =  tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_mask,
                                                        logits=self.decoder,
                                                        name='sce_loss')
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = batch_mask,
                                                          logits=self.decoder,
                                                          dim=-1,
                                                          name='softmax_loss')
        
        #td = tf.reduce_mean(tf.square(self.decoder_softmax - batch_mask))
#        print("mmse loss", td.shape)
        self.loss = tf.reduce_mean(loss)


    def create_opt(self):
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                     name='adam_opt').minimize(self.loss)

        self.opt= opt


    def normalize_images(self, images):

        ymax = images.max(axis=(1,2), keepdims=True)
        ymin = images.min(axis=(1,2), keepdims=True)
    
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
        self.x = np.moveaxis(self.x, 3, 1)
        
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
    
    def get_patch(self, rf, rx, ry):
        data = self.x
        labels = self.y
        size = self.width
        d = size//2
        ''' get all channels'''
        data_patch = data[rf, ry - d:ry  + d,rx - d:rx + d, :].copy()
        label_patch = labels[rf, ry - d:ry  + d,rx - d:rx + d, :].copy()
        return data_patch, label_patch

    def get_batch(self, num, ones=None, zeros=None, erode=0):
        erode = 0
        data = self.x
        labels = self.y
        size = self.width
        d = size//2
        batch = np.zeros((num, size, size, data.shape[-1]), dtype=np.float32)
        mask = np.zeros((num, size, size, data.shape[-1] + 2), dtype=np.float32)

        if ones == None:
            xrand = np.random.randint(d, data.shape[2] - d, num)
            yrand = np.random.randint(d, data.shape[1] - d, num)
            frand = np.random.randint(0, data.shape[0], num)
        else:
            orand = np.random.randint(2, num-2)
            fones = ones[0]
            xones = ones[2]
            yones = ones[1]
            fzeros = zeros[0]
            xzeros = zeros[2]
            yzeros = zeros[1]
            rxy = np.random.randint(0, len(ones[0]), orand)
            xrand1 = xones[rxy]
            yrand1 = yones[rxy]
            frand1 = fones[rxy]
            rxy = np.random.randint(0, len(zeros[0]), num-orand)
            xrand0 = xzeros[rxy]
            yrand0 = yzeros[rxy]
            frand0 = fzeros[rxy]

            xrand = np.concatenate((xrand1, xrand0))
            yrand = np.concatenate((yrand1, yrand0))
            frand = np.concatenate((frand1, frand0))
            perm = np.random.permutation(len(xrand))
            xrand = xrand[perm]
            yrand = yrand[perm]
            frand = frand[perm]

        for i in range(num):
            a, b = self.get_patch(frand[i], xrand[i], yrand[i])
            #print(a.shape)
            batch[i] = a
            mask[i] = b
            '''
            for ki in range(erode):
                #print(erode, i)
        
                for ic in range(data.shape[-1] - 1):
                     b[:,:,ic] = erosion(b[:,:,ic])

            b[:,:,-1] = 0
            b[:,:,-1] = 1 - b.max(axis=-1)
            mask[i] = b
            '''
        rot = np.random.randint(0,4)
        batch = np.rot90(batch, k=rot, axes=(1,2))
        mask = np.rot90(mask, k=rot, axes=(1,2))
        return batch, mask
    ### end get_batch ###

    '''
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
    '''

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


    
