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

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

input_dir = './dataset/test/'
checkpoint_dir = './checkpoint/'
result_dir = './result/'

test_fns = glob.glob(input_dir + '*.bmp')

# model setting
in_image = tf.placeholder(tf.float32, [None, None, None, 3])
_, out_image = CBDNet(in_image)

# load model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded', checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'test/'):
    os.makedirs(result_dir + 'test/')

for ind, test_fn in enumerate(test_fns):
    print(test_fn)
    noisy_img = cv2.imread(test_fn)
    noisy_img = noisy_img[:,:,::-1] / 255.0
    noisy_img = np.array(noisy_img).astype('float32')
    temp_noisy_img = np.expand_dims(noisy_img, axis=0)

    output = sess.run(out_image, feed_dict={in_image:temp_noisy_img})
    output = np.minimum(np.maximum(output, 0), 1)

    temp = np.concatenate((temp_noisy_img[0, :, :, :], output[0, :, :, :]), axis=1)
    scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(result_dir + 'test/test_%d.jpg'%(ind))

