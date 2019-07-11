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


def load_CRF():
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

    return CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl

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
    input_dir = './dataset/synthetic/'
    checkpoint_dir = './checkpoint/synthetic/'
    result_dir = './result/synthetic/'

    save_freq = 100

    CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl = load_CRF()

    train_fns = glob.glob(input_dir + '*.bmp')

    origin_imgs = [None] * len(train_fns)
    noise_imgs = [None] * len(train_fns)

    for i in range(len(train_fns)):
        origin_imgs[i] = []
        noise_imgs[i] = []

    in_image, gt_image, gt_noise, est_noise, out_image, G_loss, lr, G_opt = model_setting()

    # load model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    save_vars = [v for v in tf.global_variables() if (v.name.split('/')[0] == 'fcn' or v.name.split('/')[0] == 'unet')]
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        print('loaded', checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

    allpoint = glob.glob(checkpoint_dir + 'epoch-*')
    lastepoch = 0
    for point in allpoint:
        cur_epoch = re.findall(r'epoch-(\d+)', point)
        lastepoch = np.maximum(lastepoch, int(cur_epoch[0]))

    learning_rate = 1e-4
    for epoch in range(lastepoch, 201):
        losses = AverageMeter()

        if os.path.isdir(result_dir+"%04d"%epoch):
            continue    
        cnt=0
        
        if epoch > 100:
            learning_rate = 1e-5

        for ind in np.random.permutation(len(train_fns)):
            train_fn = train_fns[ind]

            if not len(origin_imgs[ind]):
                origin_img = ReadImg(train_fn)
                origin_imgs[ind] = np.expand_dims(origin_img, axis=0)

            # re-add noise
            if epoch % save_freq == 0:
                noise_imgs[ind] = []

            if len(noise_imgs[ind]) < 1:
                noise_img = AddRealNoise(origin_imgs[ind][0, :, :, :], CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl)
                noise_imgs[ind].append(np.expand_dims(noise_img, axis=0))

            st = time.time()
            for nind in np.random.permutation(len(noise_imgs[ind])):
                temp_origin_img = origin_imgs[ind]
                temp_noise_img = noise_imgs[ind][nind]
                temp_origin_img, temp_noise_img = DataAugmentation(temp_origin_img, temp_noise_img)
                noise_level = temp_noise_img - temp_origin_img

                cnt += 1
                st = time.time()

                _, G_current, output = sess.run(
                    [G_opt, G_loss, out_image], 
                    feed_dict={in_image:temp_noise_img, gt_image:temp_origin_img, gt_noise:noise_level, lr:learning_rate}
                    )
                output = np.clip(output, 0, 1)
                losses.update(G_current)

                print("%d %d Loss=%.4f Time=%.3f"%(epoch, cnt, losses.avg, time.time()-st))
                
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
