import tensorflow as tf
import numpy as np

def sampling(z_mean, z_log_sigma, batchsize, length):
    eps = tf.random_normal(shape=(batchsize, length), mean=0., stddev=4.)
    return z_mean + tf.exp(z_log_sigma) * eps


def leaky_relu(x, alpha=0.2):
    """
    Calculate activation with leaky Relu
    Parameters
    ----------
    x : tensor
    alpha : float

    Returns
    -------
    Activation value
    """
    return tf.maximum(x * alpha, x)


def clipped_u(x, clip=2.):
    return tf.clip_by_value(x, -clip, clip)


def get_init(stdev):
    """
    Create a layer initializer
    Returns
    -------
    Normal initializer
    """
    return tf.truncated_normal_initializer(stddev=stdev) #.04


def get_reg():
    return tf.contrib.layers.l2_regularizer(.01)


def dud():
    def __enter__():
        return 1

    def exit():
        pass

def dropout(x, is_train, rate):
    """
    Perform dropout on the input tensor
    Parameters
    ----------
    x : tensor
        Input tensor
    is_train : boolean
        True if training
    rate : float
        Keep probability

    Returns
    -------

    """
    if is_train:
        x = tf.nn.dropout(x, rate)
    return x

def layer_conv2d(x, nfilters, size, strides, padding, name, stdev,
                 droprate, is_train, activation=None):
    """
    Create a 2D convolutional layer
    
    Parameters
    ----------
    x : Tensor
    nfilters : int
        Number of filter layer
    size: int
        Size of the filter kernel
    strides : int
        Stride length of the kernel
    padding : str
        How to do the padding on the edges
    name : str
        name of the tensorflow variable
    droprate : float
        keep probability
    is_train : boolean
        True is doing training
    activation : function
        activation function

    Returns
    -------
    z : tensor
    """

    z = tf.layers.conv2d(x, nfilters, size, strides=strides,
                         padding=padding, kernel_initializer=get_init(stdev),
                         name=name, activation=activation)
    z = leaky_relu(z)
    z = dropout(z, is_train, droprate)
    return z


def encoder(images, latent_size, droprate=0.7, is_train=True,
            nfilters=None, stdev=0.04, knum=None, channel=None, denoise=None):
    """
    Build the encoder part of th neural network
    
    Parameters
    ----------
    images : numpy.array
        batch of images
    latent_size : int
        size of the final layer
    droprate : float
        keep probability
    is_train : boolean
        running as training or inference
    nfilters : list
        list of (stack size, filter size)

    Returns
    -------
    he : Tensor
        fully connected layer of size (batchsize, layer_layer_size)
        
    """
    
    if denoise:
        images = tf.nn.dropout(images, 1 - denoise)
        
    if knum is None:
        sknum = ""
        image = images
    else:
        sknum = "_" + str(knum)
        image = images[:, :, :, knum]
        image = tf.expand_dims(image, 3)
    print('Encoder', is_train)
    # images = tf.placeholder(tf.float32, (None, height, width, nchannels))

    print(image, image.shape)
    """create the model using the images"""
    if nfilters is None:
        k = 64 * np.asarray([1, 2, 4, 8], dtype=np.int32)
        k = [(64, 5), (128, 3), (256, 3), (512, 3)]
    else:
        k = nfilters

    if 1 == 1:
        layers = list()
        layers.append(image)
        for i, ki in enumerate(k):
            """Use the last element on the layers list"""
            hc = layer_conv2d(layers[-1], ki[0], ki[1], 2, "same",
                               "filter{:s}_{:02d}".format(sknum, i), stdev,
                               droprate, is_train, activation=None)
            layers.append(hc)

        h = tf.contrib.layers.flatten(layers[-1])
        he = tf.layers.dense(h, latent_size, kernel_initializer=get_init(stdev),
                             activation=None,
                             name='latent_space' + sknum)
        #print(layers, he)
    print(he)
    return he


def layer_upconv(x, nfilters, size, strides, stdev,
                 padding, tname, droprate, is_train):
    """
    Create a layer for the decoder, up-scale the tensor
    
    Parameters
    ----------
    x : tensor
    nfilters : int
        number of filter layers
    size : int
        size of the filter kernel
    strides : int
        length of convolution stride
    padding : str
        Type of padding
    tname : str
        variable name
    droprate : float
        Keep probability
    is_train : boolean
        True if for training

    Returns
    -------
    z : tensor
    
    """

    z = tf.layers.conv2d_transpose(x, nfilters, size, strides=strides,
                                     padding=padding,
                                     activation=None,
                                     kernel_initializer=get_init(stdev))

    z = leaky_relu(z)
    z = dropout(z, is_train, droprate)
    return z

def decoder(z, nchannels=2, width=64, droprate=.7, is_train=True,
            nfilters=None, stdev=0.4, knum=None, cat = None):

    if knum is None:
        sknum = ""
    else:
        sknum = str(knum)
        
    if nfilters is None:
        #ks = 32 * np.asarray([1, 2, 4, 8], dtype=np.int32)
        k = [(256, 3), (128, 3), (64, 3), (32, 5)]
    else:
        k = nfilters

    ## size of the first "image"
    isize = width // int(np.exp2(len(k) - 0))
    print("isize: ", isize, width)

    #     with tf.variable_scope("decoder", reuse=(not is_train)):
    
    if cat:
        ''' go from 256 -> 512 -> 256 (or what ever) '''
        pass
    
    if 1 == 1:
        layers = list()
        dh = tf.layers.dense(z, isize * isize * k[0][0], activation=None,
                             kernel_initializer=get_init(stdev))
        dh = leaky_relu(dh)
        dh = dropout(dh, is_train, droprate)
        layers.append(dh)

        for i, ki in enumerate(k):
            if i == 0:
                dh = tf.reshape(layers[-1], (-1, isize, isize, k[0][0]))
                layers.append(dh)
            else:
                tname = "upconv_{:s}_{:02d}".format(sknum, i)
                dh  = layer_upconv(layers[-1], ki[0], ki[1], 2, stdev, "same",
                                   tname, droprate, is_train)
                layers.append(dh)

        dh0 = tf.layers.conv2d_transpose(layers[-1], nchannels, 5, strides=2,
                                         padding='same',
                                         activation=None,
                                         kernel_initializer=get_init(stdev),
                                         name='decoder_out' + sknum)
        
        #rmax = tf.reduce_max(dh0)
        #rmin = tf.reduce_min(dh0)
        
        #sdh0 = tf.nn.sigmoid(dh0)  # , name='decoder_image')
        sdh0 = tf.nn.tanh(dh0)
        #sdh0 = dh0
        #dh0 = tf.minimum(dh0, 1)
        #print("dh0", dh0.shape)
        #sdh0 = tf.nn.relu(dh0)
        #sdh0 = (dh0 - rmin)/(rmax - rmin) 
        print(sdh0)
    return sdh0

def mixture(enc_stack, nclusters):
    split = tf.split(enc_stack, nclusters)
  
    concat = tf.squeeze(tf.concat(split, axis=2), axis=0)
    print("Split", split)
    print("Concat", concat.get_shape().as_list())
    print("Stack", enc_stack)
    
    stdev = 0.005
    m1 = tf.layers.dense(concat, 4*2048, activation=None,
                          kernel_initializer=get_init(stdev))
    
    #m1 = tf.nn.dropout(m1, .5)
    m2 = tf.layers.dense(m1, 2*1024, activation=None,
                          kernel_initializer=get_init(stdev))
    
    #m2 = tf.nn.dropout(m2, .5)
        
    m3 = tf.layers.dense(m2, 2*512, activation=tf.nn.tanh,
                          kernel_initializer=get_init(stdev))
    
    #m3 = tf.nn.dropout(m3, .5)
    logits = tf.layers.dense(m3, nclusters, activation=tf.nn.tanh,
                         kernel_initializer=get_init(stdev))
    
    p = tf.nn.softmax(logits)
    return p

''' combine the channel vectors into one vector, then
run through some dense layers 
'''
def combine_channels(enc_stack, nchannels):
    split = tf.split(enc_stack, nchannels)
  
    concat = tf.squeeze(tf.concat(split, axis=2), axis=0)
    print("Split", split)
    print("Concat", concat.get_shape().as_list())
    print("Stack", enc_stack)

    stdev = 0.04
    m1 = tf.layers.dense(concat, 512, activation=None,
                          kernel_initializer=get_init(stdev))
    
    m1 = leaky_relu(m1)
    m1 = tf.nn.dropout(m1, .75)
    m2 = tf.layers.dense(m1, 256, activation=None,
                          kernel_initializer=get_init(stdev))
    #m2 = leaky_relu(m2)
    #m2 = tf.nn.dropout(m2, .75)
        
    return m2

def uncombine_channels(combined, nchannels):
    
    stdev = 0.04
    m2 =  tf.layers.dense(combined, 512, activation=None,
                          kernel_initializer=get_init(stdev))
    m2 = leaky_relu(m2)
    m2 = tf.nn.dropout(m2, .75)
    
    m1 =  tf.layers.dense(m2, 256, activation=None,
                          kernel_initializer=get_init(stdev))
    m1 = leaky_relu(m2)
    m1 = tf.nn.dropout(m2, .75)
    
    
def comb_loss(images, sdd_stack, combined, nchannels):
    losses = list()
    for i in range(nchannels):
        x = tf.expand_dims(images[:,:,:,i], -1)
        r = sdd_stack[i]
        losses.append(tf.reduce_sum(tf.square(x - r)))
    
    
    tloss = tf.convert_to_tensor(losses)
    loss = tf.reduce_mean(tloss)
    print("Comb loss", x, r, tloss, losses)
    return loss

def mix_loss(images, sdd_stack, mixp, a, b):
    
    batchsize = images.get_shape().as_list()[0]
    p_entropy = -tf.reduce_sum(mixp*tf.log(mixp), (1))
    
    ##mixp_entropy = -mixp*tf.log(mixp)
    p_batch = tf.reduce_mean(mixp, (0))
    p_batch_entropy = -p_batch*tf.log(p_batch)
    pbar_entropy = tf.reduce_sum(p_batch_entropy)
    
    pmin = tf.reduce_min(p_batch)
    pmax = tf.reduce_max(p_batch)
    
    pd = pmax - pmin
    
    print("pbar", pbar_entropy)
    rerr = tf.reduce_sum(tf.square(images - sdd_stack), (2,3, 4))
    
    xloss = -tf.reduce_mean(images * tf.log(sdd_stack + 0.00001) +
                           (1 - images) * tf.log(1 - sdd_stack + .00001))
    print("Stack", sdd_stack)
    print("Stack", xloss)
 
    print("Rerr ", tf.transpose(rerr))
    print("Mix",mixp)
    rloss = tf.reduce_sum(tf.multiply(tf.transpose(rerr), mixp), (1))
    
    rploss = tf.reduce_sum(rloss + a*p_entropy)
    print(rploss)
    
    loss1 = rploss + 100*xloss
    loss2 = loss1 - b*pbar_entropy

    loss = loss2
    return loss, rploss, p_entropy, tf.reduce_mean(rerr), tf.reduce_mean(pbar_entropy)


def ae_loss(images, sdh0):
    """
    Calculate the loss, just [input image] - [decoded image]
    Parameters
    ----------
    images : tensor
        Input images
    sdh0 : tensor
        Decoded images
    
    Returns
    -------
    loss : float
        The reduced loss over all samples
    """
    xloss = -tf.reduce_mean(images * tf.log(sdh0 + 0.00001) +
                           (1 - images) * tf.log(1 - sdh0 + .00001))

    image_entropy = tf.reduce_mean(-tf.abs(images)*tf.log(tf.abs(0.0001 + images)))
    sdh0_entropy = tf.reduce_mean(-tf.abs(sdh0)*tf.log(tf.abs(0.0001 + sdh0)))
    diff = np.abs(image_entropy - sdh0_entropy)
    
    rloss = tf.reduce_sum(tf.square(images - sdh0), (1, 2, 3))
    loss = tf.reduce_mean(rloss) + 2*xloss
    print(image_entropy, sdh0_entropy, loss)
    return loss, xloss, diff

def sparse_loss(images, sdh0, h, slam):
#     xloss = -tf.reduce_mean(images * tf.log(sdh0 + 1e-6) +
#                            (1 - images) * tf.log(1 - sdh0 + 1e-6), (1, 2, 3))
    
#     xloss = tf.reduce_mean(xloss)
    rloss = tf.reduce_sum(tf.square(images - sdh0), (1, 2, 3))
    rloss = tf.reduce_mean(rloss)
    omega = slam*tf.reduce_sum(tf.abs(h))
    
#    rc1 = sdh0[:,:,:,0]
#    rc3 = sdh0[:,:,:,2]
    
#    r3loss = tf.reduce_sum(tf.square(rc3 - rc1), (1,2))
#    r3loss = tf.reduce_mean(r3loss)
    #return rloss + .01*xloss + omega, rloss, omega
    return rloss + omega, rloss, omega
    
def model_opt(loss, learning_rate):
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=.5).minimize(loss)
    return opt


