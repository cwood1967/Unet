import time
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import autoencoder.network as network
import autoencoder.utils as utils

from sklearn.cluster import KMeans

from matplotlib import pyplot as plt


class adversarial_autoencoder():

    def __init__(self, params):
        self.width = params['width']
        self.height = params['height']
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
        self.denoise = params['denoise']
        self.slam = params['slam']


    def d_initializer(self, stdev):
        """
        Create a layer initializer
        Returns
        -------
        Normal initializer
        """
        return tf.truncated_normal_initializer(stddev=stdev) #.04

    
    def create_encoder(self, images, is_train, reuse=False):

        with tf.variable_scope("encoder", reuse=reuse):
            encoder = network.encoder(images, self.latent_size,
                                       self.droprate, is_train=is_train,
                                       nfilters=self.enc_sizes, stdev=self.stdev,
                                       denoise=None)

        return encoder
    

    def create_decoder(self, sample_z,is_train, reuse=False):

        z = sample_z
        with tf.variable_scope("decoder", reuse=reuse):
            decoder= network.decoder(z, nchannels=self.nchannels,
                                     width=self.width, droprate=self.droprate,
                                     is_train=is_train, nfilters=self.dec_sizes,
                                     stdev=self.stdev)
        return decoder
    
    
    def create_discriminator(self, z, reuse=False):

        with tf.variable_scope("discriminator", reuse=reuse):
            h1 = tf.layers.dense(z, 1000,
                                 kernel_initializer=self.d_initializer(self.stdev),
                                 activation=None,
                                 name="discrim01")

            h1 = network.leaky_relu(h1)

            h2 = tf.layers.dense(h1, 1000,
                                 kernel_initializer=self.d_initializer(self.stdev),
                                 activation=None,
                                 name="discrim02")

            h2 = network.leaky_relu(h2)

            last = tf.layers.dense(h2, 1,
                                 kernel_initializer=self.d_initializer(self.stdev),
                                 activation=None,
                                 name="discrim03")

            return last
        

    def create_sample(self, size, mu, sigma):

        sample = np.random.normal(mu, sigma, size=size)
        return sample

    def reconstruction_loss(self, images, decoder):

        r1 = tf.reduce_sum(tf.square(images- decoder), axis=(1,2,3))
        rloss = tf.reduce_mean(r1)
        self.rloss = rloss
        #return rloss

    
    def discriminator_loss(self, sample_z, encoder):

        smooth = 0.2
        sample_logits = self.create_discriminator(sample_z)
        ae_logits = self.create_discriminator(encoder, reuse=True)

        sample_labels = (1 - smooth)*tf.ones_like(sample_logits)
        ae_labels = smooth + tf.zeros_like(ae_logits)

        # trick the discriminator 
        gen_labels = (1 - smooth)*tf.ones_like(ae_logits)
        sce_sample =tf.nn.sigmoid_cross_entropy_with_logits(logits=sample_logits,
                                                            labels=sample_labels)
        d_sample_loss = tf.reduce_mean(sce_sample)
        sce_ae = tf.nn.sigmoid_cross_entropy_with_logits(logits=ae_logits,
                                                         labels=ae_labels)
        d_ae_loss = tf.reduce_mean(sce_ae)
        d_loss = d_sample_loss + d_ae_loss

        sce_gen = tf.nn.sigmoid_cross_entropy_with_logits(logits=ae_logits,
                                                          labels=gen_labels)
        gen_loss = tf.reduce_mean(sce_gen)
        
        self.d_loss = d_loss
        self.gen_loss = gen_loss

    def loss(self, images, sample_z, gen_z):
        rloss = self.reconstruction_loss(images)
        dloss = self.discriminator_loss(sample_z, gen_z)

        loss = rloss + dloss
        self.rloss = rloss
        self.dloss = dloss
        self.loss = loss
        return loss
        

    def opt(self):
        tvar = tf.trainable_variables()
        encvars = [a for a in tvar if 'encoder' in a.name]
        decvars = [a for a in tvar if 'decoder' in a.name]
        dvars = [a for a in tvar if 'discriminator' in a.name]
        aevars = encvars + decvars

        for v in aevars:
            print(v)
        self.ae_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.rloss, var_list=aevars)
        self.d_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.d_loss, var_list=dvars)
        self.g_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.gen_loss, var_list=encvars)

        return self.ae_opt, self.d_opt, self.g_opt

''' end of adversarial_autoencoder class'''

class training():
    def __init__(self, params, datadir, title):
        if params is None:
            self.params = self.setup()
        else:
            self.params = params

        self.datadir = datadir
        self.title = title
        self.mmdict, self.n_all_images = self.create_mmdict(self.datadir)
        self.df = self.create_df(self.mmdict)

        tf1 = time.strftime("%Y-%m-%d-%H-%M-%S")
        tf2 = time.strftime("checkpoint-%Y-%m-%d-%H-%M-%S")
 
        savedir = '/media/cjw/Data/cyto/Checkpoints/' + tf1 + "_" + title + "/"
        savedir += tf2 + '/'
        self.logdir = savedir + 'log/'
        self.savename = savedir +  "autoencoder-{:d}x".format(params['latent_size'])
        print("Using data from:", self.datadir)
        print("Saving checkpoints to:", self.savename)
 
        if not os.path.exists(savedir):
            os.makedirs(savedir)
           

    def train(self, gpu=True, display=False, display_int=100,
              report_int=100, dfunc='idisplay',niterations=1000):

        self.display_int = display_int
        self.display = display
        self.report_int = report_int
        
        tf.reset_default_graph()
        w = self.params['width']
        self.images = tf.placeholder(tf.float32,
                                     (None, w, w, self.params['nchannels'])) 
        self.sample_z = tf.placeholder(tf.float32,
                                       (None, self.params['latent_size']))

        self.vn = adversarial_autoencoder(self.params)
        self.encoder = self.vn.create_encoder(self.images, True)
        self.decoder = self.vn.create_decoder(self.encoder, True)
        self.vn.reconstruction_loss(self.images, self.decoder)
        self.vn.discriminator_loss(self.sample_z, self.encoder)
        self.ae, self.d, self.g = self.vn.opt()

        tf.summary.scalar('rec_loss', self.vn.rloss)
        tf.summary.scalar('disc_loss', self.vn.d_loss)
        tf.summary.scalar('gen_loss', self.vn.gen_loss)
        tf.summary.histogram('distribution', self.encoder)
        
        merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()

        if gpu is False:
            config = tf.ConfigProto(
                device_count = {'GPU': 0})
        else:
            config = tf.ConfigProto(
                device_count = {'GPU': 1})
            
        self.sess = tf.Session(config=config)
        
        logwriter = tf.summary.FileWriter(self.logdir, self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())

        x1 = utils.get_sample(self.mmdict, self.df,
                                       64, w, self.params['nchannels'],
                                       channels=self.params['channels'])

        x1max = x1.max(axis=(1,2), keepdims=True)
        x1min = x1.min(axis=(1,2), keepdims=True)
        x1 = (x1 - x1min)/(x1max - x1min)
#         xn = x1 - x1.mean(axis=(1,2), keepdims=True)
#         xs = xn.std(axis=(1,2), keepdims=True)
#         xns = xn/xs
        self.test_images = x1 #xns
        #print(x1.shape, xn.shape, xs.shape, xns.shape)
        ri = 0
        for i in range(niterations):
            
            b1 = self.get_batch(self.params['batchsize'])
            bmax = b1.max(axis=(1,2), keepdims=True)
            bmin = b1.min(axis=(1,2), keepdims=True)

            b1 = (b1 - bmin)/(bmax - bmin)
#             bn = b1 - b1.mean(axis=(1,2), keepdims=True)
#             bs = bn.std(axis=(1,2), keepdims=True)
#             bns = bn/bs 
            
            batch_images = b1 #bns #self.get_batch(self.params['batchsize'])
            
            batch_z = np.random.normal(0, 1,
                                       size=(self.params['batchsize'],
                                             self.params['latent_size']))

            self.sess.run(self.ae, feed_dict={self.images:batch_images})
            self.sess.run(self.d,
                          feed_dict={self.images:batch_images,
                                     self.sample_z:batch_z})
            self.sess.run(self.g,
                          feed_dict={self.images:batch_images,
                                     self.sample_z:batch_z})
            self.sess.run(self.ae,
                          feed_dict={self.images:batch_images})

            
            if i % self.report_int == 0:
                xd = self.vn.d_loss.eval({self.images:batch_images,
                                          self.sample_z:batch_z},
                                         session=self.sess)
                xg = self.vn.gen_loss.eval({self.images:batch_images,
                                            self.sample_z:batch_z},
                                           session=self.sess)
                xr = self.vn.rloss.eval({self.images:batch_images,
                                         self.sample_z:batch_z},
                                        session=self.sess)

                summary = self.sess.run(merged,
                                        feed_dict={self.images:batch_images, self.sample_z:batch_z})
                logwriter.add_summary(summary, i)               
                print(i, xd, xg, xr)
                
                
            if self.display and i % self.display_int == 0:
                print("display")
                test_image = np.expand_dims(self.test_images[ri],axis=0)
                ri += 1
                if ri == len(self.test_images):
                    ri = 0

                if dfunc=='idisplay':
                    self.idisplay(test_image, batch_z)
                elif dfunc=='idisplay3':
                    self.idisplay3(test_image, batch_z)
                else:
                    pass
                
            if i % 1000 == 0:
                self.saver.save(self.sess, self.savename, global_step=i)

        print("Done")
                                    
    def get_batch(self, n):
        batch_images = utils.get_sample(self.mmdict, self.df, n,
                                self.params['width'], 
                                self.params['nchannels'],
                                channels=self.params['channels'])
        return batch_images

    def idisplay3(self, test_image, batch_z):
        encoded = self.encoder.eval({self.images:test_image}, session=self.sess)
        decoded = self.decoder.eval({self.encoder:encoded}, session=self.sess)
        decoded = np.squeeze(decoded)
        xspace =  self.encoder.eval({self.images:
                                     self.get_batch(self.params['batchsize'])},
                                    session=self.sess)


        plt.figure(figsize=(8,4))
        nc = self.params['nchannels']
        
        plt.subplot(2, 2, 1)
        
        di = np.squeeze(test_image)
        de = decoded
        plt.imshow(di)
        plt.subplot(2, 2, 2)
        plt.imshow(de)
        
        plt.subplot(2, 2,3)
        plt.hist(batch_z.reshape((-1)), bins=25)
        plt.subplot(2,2,4)
        plt.hist(xspace.reshape((-1)), bins=25)
        plt.show()

        
    def idisplay(self, test_image, batch_z):
        encoded = self.encoder.eval({self.images:test_image}, session=self.sess)
        decoded = self.decoder.eval({self.encoder:encoded}, session=self.sess)
        decoded = np.squeeze(decoded)
        xspace =  self.encoder.eval({self.images:
                                     self.get_batch(self.params['batchsize'])},
                                    session=self.sess)

        inum = 1
        plt.figure(figsize=(8,2))
        nc = self.params['nchannels']
        for i in range(nc):
            plt.subplot(2, nc + 2, inum)
            if nc == 1:
                di = np.squeeze(test_image)
                de = decoded
            else:
                di = np.squeeze(test_image)[:,:, i]
                de = decoded[:,:,i]
            plt.imshow(di)
            plt.subplot(2, nc + 2, inum + nc + 2)
            plt.imshow(de)
            inum += 1
        
        plt.subplot(2, 3, 3)
        plt.hist(batch_z.reshape((-1)), bins=25)
        plt.subplot(2,3,6)
        plt.hist(xspace.reshape((-1)), bins=25)
        plt.show()
        
    def setup(self):
        esize = [(128,3), (256, 3), (512,3)]
        dsize = list(reversed(esize))

        params =dict()

        params['width'] = 32
        params['height'] = 32
        params['nchannels'] = 4
        params['channels'] = [0,1,3,4]
        params['nepochs'] = 20
        params['batchsize'] = 256
        params['learning_rate'] = 0.0003
        params['restore'] = False
        params['latent_size'] = 64
        params['enc_sizes'] = esize 
        params['dec_sizes'] = dsize
        params['droprate'] = 0.85
        params['stdev'] = 0.04
        params['denoise'] = False
        params['slam'] = 0

        return params


    def create_df(self, mmdict):
        idx = 0
        dataframes = list()
        for key in mmdict.keys():
            mm = mmdict[key]
            n = mm.shape[0]
            w = mm.shape[1]
            file = n*[key[0:-3]]
            fid = range(n)
            mmfile = n*[key]
            yc = n*[w//2]
            xc = n*[w//2]
            ids = np.arange(idx, idx + n, 1) #all_ids[idx:idx + n]
            idx += n
            df = pd.DataFrame({'id':ids, 'fid':fid, 'file':file, 'mmfile':mmfile,
                              'yc':yc, 'xc':xc})

            dataframes.append(df)

        p_df = pd.concat(dataframes, ignore_index=True)
        return p_df

    def create_mmdict(self, datadir):
        mmdict = dict()
        mmfiles = utils.list_mmfiles(datadir)
        n_all_images = 0

        for mmfilename in mmfiles:
            mmheader = np.memmap(mmfilename, dtype="int32", mode='r',
                                 shape=(4,))

            header_shape = mmheader.shape
            xshape = [mmheader[0], mmheader[1], mmheader[2], mmheader[3]]
            xshape = tuple(xshape)
            del mmheader
            n_all_images += xshape[0]

            m3 = np.memmap(mmfilename, dtype='float32', offset=128,
                      mode='r', shape=xshape)
            key = mmfilename.split("/")[-1]
            mmdict[key] = m3

        return mmdict, n_all_images

''' done with training class'''


class aae_loader():

    def __init__(self, params, datadir, checkpointfile, checkpointdir):
        self.params = params 
        self.checkpointfile = checkpointfile
        self.checkpointdir = checkpointdir
        self.datadir = datadir
        self.mmdict, self.n_all_images = self.create_mmdict(self.datadir)
        self.df = self.create_df(self.mmdict)
        
    def load(self):

        w = self.params['width']
        self.images = tf.placeholder(tf.float32,
                                     (None, w, w, self.params['nchannels']))

        self.sample_z = tf.placeholder(tf.float32,
                                       (None, self.params['latent_size']))
        self.vn = adversarial_autoencoder(self.params)
        self.encoder = self.vn.create_encoder(self.images, True)
        self.decoder = self.vn.create_decoder(self.encoder, True)
        self.vn.reconstruction_loss(self.images, self.decoder)
        self.vn.discriminator_loss(self.sample_z, self.encoder)
        self.ae, self.d, self.g = self.vn.opt()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, self.checkpointfile)

    def create_df(self, mmdict):
        idx = 0
        dataframes = list()
        for key in mmdict.keys():
            mm = mmdict[key]
            n = mm.shape[0]
            w = mm.shape[1]
            file = n*[key[0:-3]]
            fid = range(n)
            mmfile = n*[key]
            yc = n*[w//2]
            xc = n*[w//2]
            ids = np.arange(idx, idx + n, 1) #all_ids[idx:idx + n]
            idx += n
            df = pd.DataFrame({'id':ids, 'fid':fid, 'file':file, 'mmfile':mmfile,
                              'yc':yc, 'xc':xc})

            dataframes.append(df)

        p_df = pd.concat(dataframes, ignore_index=True)
        return p_df

    def create_mmdict(self, datadir):
        mmdict = dict()
        mmfiles = utils.list_mmfiles(datadir)
        n_all_images = 0

        for mmfilename in mmfiles:
            mmheader = np.memmap(mmfilename, dtype="int32", mode='r',
                                 shape=(4,))

            header_shape = mmheader.shape
            xshape = [mmheader[0], mmheader[1], mmheader[2], mmheader[3]]
            xshape = tuple(xshape)
            del mmheader
            n_all_images += xshape[0]

            m3 = np.memmap(mmfilename, dtype='float32', offset=128,
                      mode='r', shape=xshape)
            key = mmfilename.split("/")[-1]
            mmdict[key] = m3

        return mmdict, n_all_images
        
''' done with reload class '''

def train(niterations, datadir=None, params=None,
          display=False, display_int=100, report_int=100, title="train"):


    if datadir is None:
        datadir = '/media/cjw/Data/cyto/mmCompensatedTifs/'

    trainer = training(params, datadir, title=title)

    #os.mkdir(savedir)
    
    w = params['width']

    tf.reset_default_graph()

    images = tf.placeholder(tf.float32, (None, w, w, params['nchannels'])) 
    sample_z = tf.placeholder(tf.float32, (None, params['latent_size']))
    
    vn = adversarial_autoencoder(params)

    encoder = vn.create_encoder(images, True)
    decoder =vn.create_decoder(encoder, True)
    #vn.create_discriminator(sample_z)
    vn.reconstruction_loss(images, decoder)
    vn.discriminator_loss(sample_z, encoder)
    ae, d, g = vn.opt()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step_counter = 0
        for i in range(niterations):
            #print(i)
            batch_images = utils.get_sample(mmdict, df, params['batchsize'],
                                            params['width'], 
                                            params['nchannels'],
                                            channels=params['channels'])
            
            batch_z = np.random.normal(0, 5, size=(params['batchsize'],
                                                   params['latent_size']))
            ##batch_z = np.random.uniform(-10, 10, size=(params['batchsize'],
            ##                                       params['latent_size']))
            sess.run(ae, feed_dict={images:batch_images})
            sess.run(d, feed_dict={images:batch_images, sample_z:batch_z})
            sess.run(g, feed_dict={images:batch_images, sample_z:batch_z})
            sess.run(ae, feed_dict={images:batch_images})

            step_counter += 1
            if i % report_int == 0:
                xd = vn.d_loss.eval({images:batch_images, sample_z:batch_z})
                xg = vn.gen_loss.eval({images:batch_images, sample_z:batch_z})
                xr = vn.rloss.eval({images:batch_images, sample_z:batch_z})
                test_image = np.expand_dims(batch_images[23],axis=0)
                encoded = encoder.eval({images:test_image})
                decoded = decoder.eval({encoder:encoded})
                decoded = np.squeeze(decoded)
                xspace =  encoder.eval({images:batch_images})
                print(i, xd, xg, xr)
                
            if display and i % display_int == 0:
                plt.figure(figsize=(8,2))
                plt.subplot(2,6,1)
                plt.imshow(np.squeeze(test_image)[:,:,0])
                plt.subplot(2,6,2)
                plt.imshow(np.squeeze(test_image)[:,:,1])
                plt.subplot(2,6,3)
                plt.imshow(np.squeeze(test_image)[:,:,2])
                plt.subplot(2,6,4)
                plt.imshow(np.squeeze(test_image)[:,:,3])
                plt.subplot(2,6,7)
                plt.imshow(decoded[:,:,0])
                plt.subplot(2,6,8)
                plt.imshow(decoded[:,:,1])
                plt.subplot(2,6,9)
                plt.imshow(decoded[:,:,2])
                plt.subplot(2,6,10)
                plt.imshow(decoded[:,:,3])
                plt.subplot(2,3,3)
                plt.hist(batch_z.reshape((-1)), bins=25)
                plt.subplot(2,3,6)
                plt.hist(xspace.reshape((-1)), bins=25)
                plt.show()
                
            if i % 1000 == 0:
                saver.save(sess, savename, global_step=step_counter)
                
        saver.save(sess, savename, global_step=step_counter)  
        

    print("Done")
    return {'ae':ae, 'encoder':encoder, 'decoder':decoder,
            'session':sess, 'saver':saver}


def cluster(nclusters, trained, niterations,
            display=False, display_int=100, report_int=100, title="cluster"):
    
    images = tf.placeholder(tf.float32, (None, w, w, params['nchannels'])) 
    sample_z = tf.placeholder(tf.float32, (None, params['latent_size']))

    ## need to do clustering using KMeans on all of the latent spaces
    ## or just a large random sample

#    cluster_batch = utils.get_sample(mmdict, 
    kmeans = KMeans(nclusters=nclusters, n_init=20)
    for i in range(niterations):

        batch_images = utils.get_sample(mmdict, df, params['batchsize'],
                                        params['width'], 
                                        params['nchannels'],
                                        channels=params['channels'])

        
        ## do the training on autoencoder
        sess.run(ae, feed_dict={images:batch_images})
        
