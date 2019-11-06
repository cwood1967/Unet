#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Unet/dnet'))
	print(os.getcwd())
except:
	pass

#%%
import time
import sys

sys.path.append('/home/cjw/Code/Unet')
#sys.path.append('/media/cjw/PythonLib/Unet')
from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_dilation
from dnet import dnet2d
from unet import unet2d

#get_ipython().run_line_magic('matplotlib', 'inline')


#%%
import tensorflow as tf


#%%
params = dict()

nf = 4
params['width'] = 256
params['height'] = 256
params['nchannels'] = 1
params['channels'] = [0]
params['nepochs'] = [1]
params['batchsize'] = [512]
params['learning_rate'] = 0.001
params['restore'] = False
params['latent_size'] = 1
params['net_sizes'] = [[nf, 1, 1], [nf, 3, 1], [nf, 3, 2],
                       [nf, 3, 4], [nf, 3, 6], [nf, 3, 8],
                       #[nf, 3, 12], [nf,  3, 18], [nf, 3, 24],
                       #[nf, 3, 30]], [nf, 3, 36], [nf, 3, 42],
                       [3, 1, 1]]

params['enc_sizes'] = [[32,5, 2], [64, 3, 2],
                       [128, 3, 2]]

params['dec_sizes'] = [[64, 3, 2], [32, 3, 2], [3, 5, 2]]

params['droprate'] = 0.1
params['stdev'] = 0.04


#%%

''' pombe '''
## read in the data
##data_tif = imread('/scratch/cjw/Data/sez/NDExp.tif')
##labels_tif = imread('/scratch/cjw/Data/sez/NDExp_labels.tif')
data_tif = np.load('/scratch/cjw/Data/sez/Rois/basic_stack.npy')
print(data_tif.shape)
labels_tif = np.load('/scratch/cjw/Data/sez/Rois/basic_mask.npy')
print(labels_tif.shape)
data_tif = data_tif[:,:,:, [0]]
print(labels_tif.shape,data_tif.shape)

plt.imshow(labels_tif[0,:,:,1])
''' end pombe '''

#%%
''' begin em 
all_data_tif = imread('/ssd1/cjw/Data/EM/Training.tif_RotShift.tif')
all_data_tif = np.moveaxis(all_data_tif, 1, -1)
all_data_tif.shape


data_tif = all_data_tif[:,:, :,  [0]]
labels_tif = all_data_tif[:, :, :, [1]]
labels_tif = binary_dilation(labels_tif).astype(labels_tif.dtype)

end of em'''


''' don't always do this
rtemp = np.zeros(labels_tif.shape[0:3] + (2,))
rtemp[:,:,:,0] = labels_tif[:,:,:,0] + labels_tif[:,:,:,2]
rtemp[:,:,:,1] = labels_tif[:,:,:,1] + labels_tif[:,:,:,3]
labels_tif = rtemp
#labels_tif = np.moveaxis(labels_tif, 1, -1)
'''

#create the background channel for the labels
labshape = labels_tif.shape
temp = np.zeros(labels_tif.shape[0:3] + (labshape[-1] + 1,), dtype=np.float32)
temp[:,:, :, 0:-1] = labels_tif
tsum = labels_tif.max() - labels_tif.max(axis=(-1))
temp[:,:,:, -1] = tsum
labels_tif = temp

labels_tif.shape
#normalize images 0-1
#try normalizing wit zero mean and equal sdev

#%%
dmin = data_tif.min(axis=(1,2),keepdims=True)
dmax = data_tif.max(axis=(1,2),keepdims=True)
dmean = data_tif.mean(axis=(1,2),keepdims=True)
dstd = data_tif.std(axis=(1,2),keepdims=True)

#data = (data_tif - dmin)/(dmax - dmin)
data = (data_tif - dmean)/dstd
labels = (labels_tif - labels_tif.min())/(labels_tif.max() - labels_tif.min())

''' don't always do this either
#put cells with both channels into channel 0
lsum = labels.sum(axis=(-1))
s2 = lsum > 1
s2 = s2.astype(np.float32)
labels[:,:,:,1] -= s2 
'''
data_tif.shape, data.shape, labels.shape, labels_tif.shape

#%%
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.imshow(data[2, :, :, 0])
plt.subplot(1,2,2)
plt.imshow(labels[2, :, :, :])
#%%

tf.reset_default_graph()
#u = dnet2d.dnet2d(params)
u = dnet2d.dnet2d(params)
u.x = data
u.y = labels


#%%
#test_images = imread('/scratch/cjw/Data/sez/test.tif')
#tmean = test_images.mean(axis=(1,2),keepdims=True)
#tstd = test_images.std(axis=(1,2),keepdims=True)

#data = (data_tif - dmin)/(dmax - dmin)
#test_images = (test_images - tmean)/tstd

u.xtest = u.x[-1:]
u.ytest = u.y[-1:]
u.x = u.x[:-1]
u.y = u.y[:-1]
#print(test_images.std(axis=(1,2)))
#u.xtest.shape, u.xtest.mean(axis=(1,2)).shape,u.xtest.mean(axis=(1,2)), u.xtest.min(axis=(1,2)), u.xtest.max(axis=(1,2))


#%%
w = 256


#%%
tf.reset_default_graph()
images = tf.placeholder(tf.float32, (None, w, w, params['nchannels']))
masks = tf.placeholder(tf.float32, (None, w, w, labels.shape[-1]))
learning_rate = tf.placeholder(tf.float32, ())
u.learning_rate = learning_rate


#%%
enc = u.create_dnet(images, True)

#enc = u.create_encoder(images, True)
#d = u.create_decoder()

#%%

u.create_loss(masks)
u.create_opt()

#%%
#u.net, labels.shape


#%%
#d = u.create_decoder()
#u.create_loss(masks)
#u.create_opt()


 #%%
#u.set_validation(20)
u.xtrain = u.x


#%%
sess = tf.Session()

#%%
ich0 = tf.expand_dims(images[:,:,:,0], axis=-1)
#ich1 = tf.expand_dims(images[:,:,:,1], axis=-1)

smch0 = tf.expand_dims(u.decoder_sigmoid[:,:,:,0], axis=-1)
smch1 = tf.expand_dims(u.decoder_sigmoid[:,:,:,1], axis=-1)

mch0 = tf.expand_dims(masks[:,:,:,0], axis=-1)
mch1 = tf.expand_dims(masks[:,:,:,1], axis=-1)

ich0, mch0, mch1, masks
#%%
# tf.summary.scalar('loss', u.loss)
# si = tf.slice(u.decoder, [5, 0, 0, 0, 0], [10,1,256,256, 1])

# si = si[:,:,:,:,0]
# print(si)
# tf.summary.image('res', si, max_outputs=1)    
# merged = tf.summary.merge_all()



# logdir = "logdir"
# logwriter = tf.summary.FileWriter(logdir, sess.graph)


#%%
sess.run(tf.global_variables_initializer())

#%%
u.stdev = 0.01
#u.learning_rate = 0.0001


#%%

rate = 0.005
#loss_file = open('progress.dat', 'w', buffering=1)

tbx, tbm = u.get_batch(16, test=True, erode=0)
tbx0 = np.expand_dims(tbx[:,:,:,0], -1)
#tbx1 = np.expand_dims(tbx[:,:,:,1], -1)
tf.summary.scalar('loss', u.loss)
tf.summary.image('channel_0', ich0)
#tf.summary.image('channel_1', ich1)
tf.summary.image('sm_channel_0', smch0)
tf.summary.image('sm_channel_1', smch1)
tf.summary.image('masks', mch0)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test')

for i in range(5500):
    t1 = time.time()
    rate = np.maximum(rate*.9997, 0.000005)
    if i >= -1:
        bx, bm = u.get_batch(128, erode=0)
        bxf0 = np.expand_dims(bx[:,:,:,0], -1)
        #bxf1 = np.expand_dims(bx[:,:,:,1], -1)
    #print(bxf0.shape,bxf1.shape)
    _, res, err, m0 = sess.run([u.opt, u.decoder_softmax, u.loss, merged],
                           feed_dict={images:bx, masks:bm, learning_rate:rate,
                                     ich0:bxf0}) #, ich1:bxf1})
    
    if i % 10 == 0:
        train_writer.add_summary(m0, i)

    if i % 100 == 0:
        print(i, err, err, res[0,:,:,:].min(), res[0,:,:,:].max(), rate)
        nr = np.random.randint(0,16)
        #nr = np.argmin(res[:,:,:,4].sum(axis=(1,2)))
        print(nr)
        
        tres, tloss, m1 = sess.run([u.decoder_softmax, u.loss, merged],
                           feed_dict={images:tbx, masks:tbm, learning_rate:rate,
                                      ich0:tbx0}) #, ich1:tbx1})
        
        test_writer.add_summary(m1, i)

    t2 = time.time()
    if i % 10 == 0:
        print(i, err, (t2 - t1), rate)


 ##%%
na = 13
plt.figure(figsize=(14,6))
plt.subplot(1,3,1)
print(bm.shape)
plt.imshow(res[na,:,:,0:3])
plt.subplot(1,3,2)
plt.imshow(bm[na,:,:,0:3])
plt.subplot(1,3,3)
plt.imshow(bx[na,:,:,:])


#%%
plt.hist(res[na,:,:,:].reshape((-1,3)))

#%%
#test_tif = imread('/ssd1/cjw/Data/sez/RawTifs/16_hours_punched_overnight/NDExp_Point0008_Seq0008.tif')
test_tif = imread('/ssd1/cjw/Data/sez/RawTifs/NDExp_Point0007_Seq0007.tif')
#test_tif = labels[-1,:,:,:] #imread('/ssd1/cjw/Data/sez/RawTifs/NDExp_Point0021_Seq0021.tif')
ymax = test_tif.max(axis=(0,1), keepdims=True)
ymin = test_tif.min(axis=(0,1), keepdims=True)
ymean = test_tif.mean(axis=(0,1), keepdims=True)
ystd = test_tif.std(axis=(0,1), keepdims=True)
#test_tif = (test_tif - ymin)/(ymax - ymin + 0.00001)
test_tif = (test_tif - ymean)/ystd
test_tif.shape, ymin.shape


#%%
plt.imshow(test_tif)


#%%
def make_batch(data, start = 0):
    tclist = list()
    print(start)
    ix = start
    jy = start
    for i in range(6):
        jy = start
        for j in range(6):
            p = data[jy:jy + 256, ix:ix + 256,:]
            p = np.expand_dims(p, 0)
            tclist.append(p)
            jy = jy + 256
            #print(i, j, ix, jy)
        ix += 256

    tc = np.concatenate(tclist, axis=0)
    return tc

tc0 = make_batch(test_tif,start=0)
tc1 = make_batch(test_tif, start=128)
tc0.shape, tc1.shape


#%%
vres0 = sess.run(u.decoder_softmax,
                feed_dict={images:tc0})

vres1 = sess.run(u.decoder_softmax,
                feed_dict={images:tc1})
vres0.shape, vres1.shape


#%%
def reconstruct(v1, w, nx, ny, offset=0, pad=0):
    p = np.zeros((ny*w + pad, nx*w + pad, 5), dtype=np.float32)

    ix = offset
    jy = offset
    index = 0
    for i in range(nx):
        jy = offset
        for j in range(ny):
            p[jy:jy + 256, ix:ix + 256, :] = v1[index]
            jy = jy + 256
            index += 1
            #print(i, j, ix, jy)
        ix += 256
    return p
r0 = reconstruct(vres0, 256, 6, 6, offset=0, pad=128)
r1 = reconstruct(vres1, 256, 6, 6, offset=128, pad = 128)

r0 = np.expand_dims(r0, 0)
r1 = np.expand_dims(r1, 0)
r = np.concatenate([r0, r1])
r  = r.max(axis=(0))
r.shape, r.max()


#%%

plt.figure(figsize=(12,12))

# az = r*0
# az[:,:,0] = r[:,:,-1]
# az[:,:,1] = r[:,:,-1]
# az[:,:,2] = r[:,:,-1]
# az = r - az

#plt.subplot(1,3,1)
plt.imshow(r[:,:,0:3]) #0[0,150:350,100:300,1])

#plt.subplot(1,3,4)
#plt.imshow(r0[0]) #[0,150:350,100:300,1])

#plt.subplot(1,3,3)
#plt.imshow(r1[0]) #[150:350,100:300,1])


#%%
plt.figure(figsize=(12,12))
plt.imshow(test_tif[0:6*256+128, 0:6*256+128, :])


#%%
plt.imshow(tc0[0,:,:,0])


#%%
import tifffile
r.shape


#%%
tifffile.imsave('/ssd1/cjw/Data/sez/unetout0017-2.tif', np.moveaxis(r, -1, 0))


#%%
from skimage.morphology import binary_erosion

blab = labels[2, :,:,1]
plt.subplot(1,2,1)
plt.imshow(blab)
plt.subplot(1,2,2)
plt.imshow(binary_erosion(binary_erosion(blab)))
labels.shape


#%%
plt.imshow(data[6,:,:,0:3])


#%%
get_ipython().system("%mkdir '/ssd1/cjw/sez/unet_checkpoint'")


#%%
saver = tf.train.Saver()
saver.save(sess, '/ssd1/cjw/sez/Checkpoints/2019-02-04-zero-mean/unet_checkpoint', global_step=5000)


