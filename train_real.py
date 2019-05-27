from __future__ import division
from __future__ import print_function
import os, time, scipy.io
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import glob
import re

from utils.noise import *
from utils.common import *
from model import *


def model_setting():
    in_image = tf.placeholder(tf.float32, [None, None, None, 3])
    gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
    gt_noise = tf.placeholder(tf.float32, [None, None, None, 3])

    est_noise, out_image = CBDNet(in_image)

    G_loss = tf.losses.mean_squared_error(gt_image, out_image) + \
            0.5 * tf.reduce_mean(tf.multiply(tf.abs(0.3 - tf.nn.relu(gt_noise - est_noise)), tf.square(est_noise - gt_noise))) + \
            0.05 * tf.reduce_mean(tf.square(tf.image.image_gradients(est_noise)))

    lr = tf.placeholder(tf.float32)
    G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

    return in_image, gt_image, gt_noise, est_noise, out_image, G_loss, lr, G_opt

def DataAugmentation(temp_origin_img, temp_noise_img):
    if np.random.randint(2, size=1)[0] == 1:
        temp_origin_img = np.flip(temp_origin_img, axis=1)
        temp_noise_img = np.flip(temp_noise_img, axis=1)
    if np.random.randint(2, size=1)[0] == 1: 
        temp_origin_img = np.flip(temp_origin_img, axis=0)
        temp_noise_img = np.flip(temp_noise_img, axis=0)
    if np.random.randint(2, size=1)[0] == 1:
        temp_origin_img = np.transpose(temp_origin_img, (0, 2, 1, 3))
        temp_noise_img = np.transpose(temp_noise_img, (0, 2, 1, 3))
    
    return temp_origin_img, temp_noise_img


if __name__ == '__main__':
    input_dir = './dataset/real/'
    checkpoint_dir = './checkpoint/'
    result_dir = './result/'

    ps = 1024
    save_freq = 100

    train_fns = glob.glob(input_dir + 'Batch_*')

    origin_imgs = [None] * len(train_fns)
    noise_imgs = [None] * len(train_fns)

    for i in range(len(train_fns)):
        origin_imgs[i] = []
        noise_imgs[i] = []

    # model setting
    in_image = tf.placeholder(tf.float32, [None, None, None, 3])
    gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
    gt_noise = tf.placeholder(tf.float32, [None, None, None, 3])

    est_noise, out_image = CBDNet(in_image)

    G_loss = tf.losses.mean_squared_error(gt_image, out_image) + \
            0.5 * tf.reduce_mean(tf.multiply(tf.abs(0.3 - tf.nn.relu(gt_noise - est_noise)), tf.square(est_noise - gt_noise))) + \
            0.05 * tf.reduce_mean(tf.square(tf.image.image_gradients(est_noise)))

    lr = tf.placeholder(tf.float32)
    G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

    # load model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded', checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

    allpoint = glob.glob(checkpoint_dir+'epoch-*')
    lastepoch = 0
    for point in allpoint:
        cur_epoch = re.findall(r'epoch-(\d+)', point)
        lastepoch = np.maximum(lastepoch, int(cur_epoch[0]))

    learning_rate = 1e-4
    for epoch in range(lastepoch, 2001):
        losses = AverageMeter() 
        cnt=0
        
        if epoch > 1000:
            learning_rate = 1e-5

        for ind in np.random.permutation(len(train_fns)):
            train_fn = train_fns[ind]

            if not len(origin_imgs[ind]):
                train_origin_fns = glob.glob(train_fn + '/*Reference.bmp')
                train_noise_fns = glob.glob(train_fn + '/*Noisy.bmp')

                origin_img = ReadImg(train_origin_fns[0])
                origin_imgs[ind] = np.expand_dims(origin_img, axis=0)

                for train_noise_fn in train_noise_fns:
                    noise_img = ReadImg(train_noise_fn)
                    noise_imgs[ind].append(np.expand_dims(noise_img, axis=0))

            st = time.time()
            for nind in np.random.permutation(len(noise_imgs[ind])):
                H = origin_imgs[ind].shape[1]
                W = origin_imgs[ind].shape[2]

                ps_temp = min(H, W, ps) - 1

                xx = np.random.randint(0, W-ps_temp)
                yy = np.random.randint(0, H-ps_temp)
                
                temp_origin_img = origin_imgs[ind][:, yy:yy+ps_temp, xx:xx+ps_temp, :]
                temp_noise_img = noise_imgs[ind][nind][:, yy:yy+ps_temp, xx:xx+ps_temp, :]
                temp_origin_img, temp_noise_img = DataAugmentation(temp_origin_img, temp_noise_img)
                noise_level = temp_noise_img - temp_origin_img

                cnt += 1
                st = time.time()

                _, G_current, output = sess.run([G_opt, G_loss, out_image], feed_dict={in_image:temp_noise_img, gt_image:temp_origin_img, gt_noise:noise_level, lr:learning_rate})
                output = np.minimum(np.maximum(output, 0), 1)
                losses.update(G_current)

                print("%d %d Loss=%.5f Time=%.3f"%(epoch, cnt, losses.avg, time.time()-st))

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
