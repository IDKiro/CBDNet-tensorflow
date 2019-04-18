from __future__ import division
from __future__ import print_function
import os, time, scipy.io
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import glob
import re
import cv2

from utils import *
from model import *

input_dir = 'dataset/train/'
checkpoint_dir = './checkpoint/'
result_dir = './result/'

LEVEL = 5
save_freq = 100

CRF = scipy.io.loadmat('matdata/201_CRF_data.mat')
iCRF = scipy.io.loadmat('matdata/dorfCurvesInv.mat')
B_gl = CRF['B']
I_gl = CRF['I']
B_inv_gl = iCRF['invB']
I_inv_gl = iCRF['invI']

if os.path.exists('matdata/201_CRF_iCRF_function.mat')==0:
    CRF_para = np.array(CRF_function_transfer(I_gl, B_gl))
    iCRF_para = 1. / CRF_para
    scipy.io.savemat('matdata/201_CRF_iCRF_function.mat', {'CRF':CRF_para, 'iCRF':iCRF_para})
else:
    Bundle = scipy.io.loadmat('matdata/201_CRF_iCRF_function.mat')
    CRF_para = Bundle['CRF']
    iCRF_para = Bundle['iCRF']

train_fns = glob.glob(input_dir + '*.jpg')

origin_imgs = [None] * len(train_fns)
noise_imgs = [None] * len(train_fns)

for i in range(len(train_fns)):
    origin_imgs[i] = []
    noise_imgs[i] = []

# model setting
in_image = tf.placeholder(tf.float32, [None, None, None, 3])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

out_image = CBDNet(in_image)

G_loss = tf.losses.mean_squared_error(gt_image, out_image)
lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)
t_vars = tf.trainable_variables()

# load model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded', checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

g_loss = np.zeros((3000, 1))

allpoint = glob.glob(checkpoint_dir+'epoch-*')
lastepoch = 0
for point in allpoint:
    cur_epoch = re.findall(r'epoch-(\d+)', point)
    lastepoch = np.maximum(lastepoch, int(cur_epoch[0]))

learning_rate = 1e-4
for epoch in range(lastepoch, 2001):
    if os.path.isdir(result_dir+"%04d"%epoch):
        continue    
    cnt=0
    
    if epoch > 1000:
        learning_rate = 1e-5

    for ind in np.random.permutation(len(train_fns)):
        train_fn = train_fns[ind]

        if not len(origin_imgs[ind]):
            origin_img = cv2.imread(train_fn)
            origin_img = origin_img[:,:,::-1] / 255.0
            origin_img = np.array(origin_img).astype('float32')
            origin_imgs[ind] = np.expand_dims(origin_img, axis=0)

        # re-add noise
        if epoch % save_freq == 0:
            noise_imgs[ind] = []

        if len(noise_imgs[ind]) < LEVEL:
            for noise_i in range(LEVEL):
                sigma_s = np.random.uniform(0.0, 0.16, (3,))
                sigma_c = np.random.uniform(0.0, 0.06, (3,))
                CRF_index = np.random.choice(201)
                pattern = np.random.choice(4) + 1

                noise_img = AddNoiseMosai(origin_img, CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl, sigma_s, sigma_c, CRF_index, pattern, 0)
                noise_imgs[ind].append(np.expand_dims(noise_img, axis=0))

        st = time.time()
        for nind in np.random.permutation(len(noise_imgs[ind])):
            temp_origin_img = origin_imgs[ind]
            temp_noise_img = noise_imgs[ind][nind]
            if np.random.randint(2, size=1)[0] == 1:
                temp_origin_img = np.flip(temp_origin_img, axis=1)
                temp_noise_img = np.flip(temp_noise_img, axis=1)
            if np.random.randint(2, size=1)[0] == 1: 
                temp_origin_img = np.flip(temp_origin_img, axis=0)
                temp_noise_img = np.flip(temp_noise_img, axis=0)
            if np.random.randint(2, size=1)[0] == 1:
                temp_origin_img = np.transpose(temp_origin_img, (0, 2, 1, 3))
                temp_noise_img = np.transpose(temp_noise_img, (0, 2, 1, 3))
            
            cnt += 1
            if cnt % LEVEL == 1:
                st = time.time()

            _, G_current, output = sess.run([G_opt, G_loss, out_image], feed_dict={in_image:temp_noise_img, gt_image:temp_origin_img, lr:learning_rate})
            output = np.minimum(np.maximum(output, 0), 1)
            g_loss[ind] = G_current

            if cnt % LEVEL == 0:
                print("%d %d Loss=%.5f Time=%.3f"%(epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time()-st))

            if epoch % save_freq == 0:
                if not os.path.isdir(result_dir + '%04d'%epoch):
                    os.makedirs(result_dir + '%04d'%epoch)

                temp = np.concatenate((temp_origin_img[0, :, :, :], temp_noise_img[0, :, :, :], output[0, :, :, :]), axis=1)
                scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/train_%d_%d.jpg'%(epoch, ind, nind))
    
    saver.save(sess, checkpoint_dir + 'model.ckpt')

    if not os.path.isdir(checkpoint_dir + 'epoch-' + str(epoch)):
        os.mkdir(checkpoint_dir + 'epoch-' + str(epoch))

    if os.path.isdir(checkpoint_dir + 'epoch-' + str(epoch - 1)):
        os.rmdir(checkpoint_dir + 'epoch-' + str(epoch - 1))
