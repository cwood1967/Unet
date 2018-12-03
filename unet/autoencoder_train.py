
import sys
import warnings
from distutils.version import LooseVersion
import time
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt



# print(sys.path)
from autoencoder import network
from autoencoder import utils


def check_gpu():
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def display(sess, x1, x2, x3, x4, x5, x6, p1, p2, p3, p4):
    f = plt.figure(figsize=(12, 4))
    plt.subplot(1, 6, 1)
    plt.imshow(x1)
    plt.subplot(1, 6, 2)
    plt.imshow(x2)

    plt.subplot(1, 6, 3)
    plt.imshow(x3)
    plt.subplot(1, 6, 4)
    plt.imshow(x4)

    plt.subplot(1, 6, 5)
    plt.imshow(x5)
    plt.subplot(1, 6, 6)
    plt.imshow(x6)

    plt.show()
    plt.plot(p1)
    plt.plot(p2)
    plt.plot(p3, color='red')
    #plt.plot(p4, color='black')
    plt.show()


def showrow(fig, g, b, s, start):
    sy, sx = s
    
    counter = 0
    for i in range(g[0].shape[-1]):
        fig.add_subplot(sy,sx,start + counter)
        gx = g[:,:,i]
        mxg = gx.max()
        mng = gx.min()
        dg = (gx - mxg)/(mxg - mng)
        plt.imshow(dg)
        counter += 1
    '''
    fig.add_subplot(sy, sx, start + 1)
    plt.imshow(g[:,:,1])
    fig.add_subplot(sy, sx, start + 2)
    plt.imshow(g[:,:,2])
    '''
    
    for i in range(b[0].shape[-1]):
        fig.add_subplot(sy,sx,start + counter)
        plt.imshow(b[:,:,i])
        counter += 1
        
    '''
    fig.add_subplot(sy, sx, start + 4)
    plt.imshow(b[:,:,1])
    fig.add_subplot(sy, sx, start + 5)
    plt.imshow(b[:,:,2])
    '''
    
def display2(sess, g, b, plots):
    f = plt.figure(figsize=(12, 4))
    
    x = 1
    for i in range(len(g)):
        showrow(f, g[i], b[i], (len(g) + 1,2*g[i].shape[-1]), x)
        x += 2*g[i].shape[-1]
   
    plt.subplot(len(g) + 1, 1, len(g) + 1)
    for p in plots:
        plt.plot(p)
    plt.show()
             
''' need to have the data file, a csv file with x, y, etc
to create the dataframe
'''


def train(mmdict, df, params, ndisp, saveto=None, tloss=.001):
    """Train the autoencoder neural network and save the results
    
    Arguments:
        mmdict -- dictionary of memory map files
        params -- dictionay of parameters
    """

    tf.reset_default_graph()

    width = params['width']
    height = params['height']
    nchannels = params['nchannels']
    channels = params['channels']
    nepochs = params['nepochs']
    batchsize = params['batchsize']
    learning_rate = params['learning_rate']
    restore = params['restore']
    latent_size = params['latent_size']
    enc_sizes = params['enc_sizes']
    dec_sizes = params['dec_sizes']
    droprate = params['droprate']
    stdev = params['stdev']
    denoise = params['denoise']
    slam = params['slam']
    
    if saveto is None:
        saveto = ""
    else:
        if not saveto.endswith("/"):
            saveto += "/"
           
    savedir = saveto + time.strftime("checkpoint-%Y-%m-%d-%H-%M-%S")
    os.mkdir(savedir)
    savename = savedir + "/" + "autoencoder-{:d}x".format(latent_size)

    images = tf.placeholder(tf.float32, (None, height, width, nchannels))
    z = tf.placeholder(tf.float32, (None, latent_size))

    enc = network.encoder(images, latent_size, droprate=droprate, is_train=True,
                          nfilters=enc_sizes, stdev=stdev, denoise = denoise)
    sdd = network.decoder(enc, nchannels=nchannels, width=width,
                          droprate=droprate,
                          is_train=True, nfilters=dec_sizes, stdev=stdev)

    #loss, ient, gent = network.ae_loss(images, sdd, enc)
    loss, ient, gent = network.sparse_loss(images, sdd, enc, slam)
    
    opt = network.model_opt(loss, learning_rate)
    print(len(df), len(df)//batchsize, batchsize)
    
    shuffle_df = df.sample(len(df))
    
    #test_batch, _, _ = utils.getbatch(mmdict, df, len(df) // batchsize - 1,
    #                                  batchsize, width, nchannels,
    #                                  channels=channels)

    testdf = shuffle_df.iloc[:512]
    df = shuffle_df.iloc[512:]
    print(len(df), len(testdf))
    samp = 26

    tb0 = utils.get_sample(mmdict, testdf, len(testdf),
                                  width, nchannels,
                                  channels=channels)

    tbmean = tb0.mean(axis=(1,2), keepdims=True)
    
#     tbmax = tb0.max(axis=(1,2), keepdims=True)
#     tbmin = tb0.min(axis=(1,2), keepdims=True)
    
    test_batch = tb0 - tbmean #(tb0 - tbmin)/(tbmax - tbmin)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if restore:
            pass
        else:
            counter = 0
            
            for ib in range(len(df)*nepochs):
                start = 0
                
                #while 1==1: #start < len(df) // batchsize - 1:
                ##### using this: get_sample(mmdict, df, n, size, nchannels, channels=None):
                ####batch, wells, rownums = utils.getbatch(mmdict,
                batch = utils.get_sample(mmdict, df, batchsize,
                                         width, nchannels,
                                         channels=channels)

                bmean = batch.mean(axis=(1,2), keepdims=True)
#                 bmax = batch.max(axis=(1,2), keepdims=True)
#                 bmin = batch.min(axis=(1,2), keepdims=True)

                batch = batch = batch - bmean #(batch - bmin)/(bmax - bmin)
                #batch[:,:,:, 1] *= .5
#                     rv = .5*(np.random.rand(batchsize) + 1)
#                     rv = rv[:, np.newaxis, np.newaxis]
#                     batch[:,:,:,2] *= rv
                nr = 0 #np.random.randint(0,4)
                if nr > 0:
                    batch = np.rot90(batch, nr, (1,2)) 
                # aenc = sess.run(enc,
                #                 feed_dict={images:batch})

                asdd, aenc, az, aient, agent, _ = sess.run([sdd, enc, loss, ient, gent, opt],
                                             feed_dict={images: batch})

                ni = np.random.randint(0, batchsize)

                if ib % (ndisp/10) == 0:
                    test_he = sess.run(enc, feed_dict={images: test_batch})
                    test_sdd, test_loss = sess.run([sdd, loss], feed_dict={enc: test_he,
                                                        images: test_batch})


                    print('Iteration: ', ib, 'Loss: ', az, aient, agent)
                    print("Test Loss", test_loss)
                    if test_loss < tloss:
                        break
                        
                if ib % ndisp == 0:
                    sdh0r = asdd[ni]
#                     test_he = sess.run(enc, feed_dict={images: test_batch})
#                     test_sdd, test_loss = sess.run([sdd, loss], feed_dict={enc: test_he,
#                                                         images: test_batch})

                    print('Iteration: ', ib, 'Loss: ', az, aient, agent)
                    print("Test Loss", test_loss)
                    '''
                    display(sess, sdh0r[:, :, 0], batch[ni, :, :, 0],
                            test_sdd[ni, :, :, 0], test_batch[ni, :, :, 0],
                            test_sdd[4, :, :, 0], test_batch[4, :, :, 0],
                            test_he[ni], aenc[ni],
                            test_he[4], test_he[0])
                    '''

                    gd = [sdh0r[:, :, :], test_sdd[ni, :, :, :], test_sdd[4, :, :, :]]
                    bd = [batch[ni, :, :, :], test_batch[ni, :, :, :], test_batch[4, :, :, :]]
                    pd = [aenc[ni],  test_he[ni], test_he[4]]
                    display2(sess, gd, bd, pd)
                    
                    saver.save(sess, savename, global_step=ib)
                    '''
                    display2(sess, sdh0r[:, :, :], batch[ni, :, :, :],
                            test_sdd[ni, :, :, :], test_batch[ni, :, :, :],
                            test_sdd[4, :, :, :], test_batch[4, :, :, :])
                    '''
                
            saver.save(sess, savename, global_step=ib)
            
    print("Done")

if __name__ == '__main__':

    datadir = "/Users/cjw/Projects/cjw/yeastAE/Data/yeast/"
    datafile = datadir + "plate15_cells.csv"
    p_df = utils.read_data_file(datafile)
    mmfilename = datadir + "mmplate15-1.mm"
    mm = np.memmap(mmfilename, dtype='int32', mode='r',
                   shape=(4,))
    mmshape = mm.shape
    xshape = (mm[0], mm[1], mm[2], mm[3])
    del mm

    m2 = np.memmap(mmfilename, dtype='float32', offset=128,
                   mode='r', shape=xshape)
    p_mmdict = {"mmplate15-1.mm": m2}

    p_width = 32
    p_height = 32
    p_nchannels = 1
    p_channels = [3]
    p_nepochs = 2
    p_batchsize = 32
    p_learning_rate = 0.0004
    p_restore = False
    p_latent_size = 128

    enc_sizes = [(32, 7), (64, 5), (128, 3), (256, 3)]
    dec_sizes = list(reversed(enc_sizes))
    dec_sizes.append((1, 7))
    params = dict()

    params['width'] = p_width
    params['height'] = p_height
    params['nchannels'] = p_nchannels
    params['channels'] = p_channels
    params['nepochs'] = p_nepochs
    params['batchsize'] = p_batchsize
    params['learning_rate'] = p_learning_rate
    params['restore'] = p_restore
    params['latent_size'] = p_latent_size
    params['enc_sizes'] = enc_sizes
    params['dec_sizes'] = dec_sizes

    train(p_mmdict, p_df, params)









