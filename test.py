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

input_dir = 'dataset/synthetic/test/'
checkpoint_dir = './checkpoint/'
result_dir = './result/'

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

test_fns = glob.glob(input_dir + '*.jpg')

# model setting
in_image = tf.placeholder(tf.float32, [None, None, None, 3])
out_image = CBDNet(in_image)

# load model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded', checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

for ind, test_fn in enumerate(test_fns):
    print(test_fn)
    origin_img = cv2.imread(test_fn)
    origin_img = origin_img[:,:,::-1] / 255.0
    origin_img = np.array(origin_img).astype('float32')
    temp_origin_img = np.expand_dims(origin_img, axis=0)

    sigma_s = np.random.uniform(0.0, 0.16, (3,))
    sigma_c = np.random.uniform(0.0, 0.06, (3,))
    CRF_index = np.random.choice(201)
    pattern = np.random.choice(4) + 1

    noise_img = AddNoiseMosai(origin_img, CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl, sigma_s, sigma_c, CRF_index, pattern, 0)
    temp_noise_img = np.expand_dims(noise_img, axis=0)

    output = sess.run(out_image, feed_dict={in_image:temp_noise_img})
    output = np.minimum(np.maximum(output, 0), 1)

    temp = np.concatenate((temp_noise_img[0, :, :, :], output[0, :, :, :]), axis=1)
    scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(result_dir + 'final/test_%d.jpg'%(ind))

