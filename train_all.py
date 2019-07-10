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

    G_loss_s = tf.losses.mean_squared_error(gt_image, out_image) + \
            0.5 * tf.reduce_mean(tf.multiply(tf.abs(0.3 - tf.nn.relu(gt_noise - est_noise)), tf.square(est_noise - gt_noise))) + \
            0.05 * tf.reduce_mean(tf.square(tf.image.image_gradients(est_noise)))

    G_loss_r = tf.losses.mean_squared_error(gt_image, out_image) + \
            0.05 * tf.reduce_mean(tf.square(tf.image.image_gradients(est_noise)))

    lr = tf.placeholder(tf.float32)

    G_opt_s = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss_s)
    G_opt_r = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss_r)

    return in_image, gt_image, gt_noise, est_noise, out_image, G_loss_s, G_loss_r, lr, G_opt_s, G_opt_r

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
    input_syn_dir = './dataset/synthetic/'
    input_real_dir = './dataset/real/'
    checkpoint_dir = './checkpoint/all/'
    result_dir = './result/all/'

    PS = 512                            # patch size, if your GPU memory is not enough, modify it
    REAPET = 10
    save_freq = 100

    CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl = load_CRF()

    train_syn_fns = glob.glob(input_syn_dir + '*.bmp')
    train_real_fns = glob.glob(input_real_dir + 'Batch_*')

    origin_syn_imgs = [None] * len(train_syn_fns)
    noise_syn_imgs = [None] * len(train_syn_fns)

    origin_real_imgs = [None] * len(train_real_fns)
    noise_real_imgs = [None] * len(train_real_fns)

    for i in range(len(train_syn_fns)):
        origin_syn_imgs[i] = []
        noise_syn_imgs[i] = []

    for i in range(len(train_real_fns)):
        origin_real_imgs[i] = []
        noise_real_imgs[i] = []

    in_image, gt_image, gt_noise, est_noise, out_image, G_loss_s, G_loss_r, lr, G_opt_s, G_opt_r = model_setting()

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
    for epoch in range(lastepoch, 201):
        losses = AverageMeter()

        if os.path.isdir(result_dir+"%04d"%epoch):
            continue    
        cnt=0
        
        if epoch > 100:
            learning_rate = 1e-5

        print('Training on synthetic noisy images...')
        for ind in np.random.permutation(len(train_syn_fns)):
            train_syn_fn = train_syn_fns[ind]

            if not len(origin_syn_imgs[ind]):
                origin_syn_img = ReadImg(train_syn_fn)
                origin_syn_imgs[ind] = np.expand_dims(origin_syn_img, axis=0)

            # re-add noise
            if epoch % save_freq == 0:
                noise_syn_imgs[ind] = []

            if len(noise_syn_imgs[ind]) < 1:
                noise_syn_img = AddRealNoise(origin_syn_imgs[ind][0, :, :, :], CRF_para, iCRF_para, I_gl, B_gl, I_inv_gl, B_inv_gl)
                noise_syn_imgs[ind].append(np.expand_dims(noise_syn_img, axis=0))

            st = time.time()
            for nind in np.random.permutation(len(noise_syn_imgs[ind])):
                temp_origin_img = origin_syn_imgs[ind]
                temp_noise_img = noise_syn_imgs[ind][nind]
                temp_origin_img, temp_noise_img = DataAugmentation(temp_origin_img, temp_noise_img)
                noise_level = temp_noise_img - temp_origin_img

                cnt += 1
                st = time.time()

                _, G_current, output = sess.run([G_opt_s, G_loss_s, out_image], feed_dict={in_image:temp_noise_img, gt_image:temp_origin_img, gt_noise:noise_level, lr:learning_rate})
                output = np.clip(output, 0, 1)
                losses.update(G_current)

                print("%d %d Loss=%.4f Time=%.3f"%(epoch, cnt, losses.avg, time.time()-st))
                
                if epoch % save_freq == 0:
                    if not os.path.isdir(result_dir + '%04d'%epoch):
                        os.makedirs(result_dir + '%04d'%epoch)

                    temp = np.concatenate((temp_origin_img[0, :, :, :], temp_noise_img[0, :, :, :], output[0, :, :, :]), axis=1)
                    scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/train_%d_%d.jpg'%(epoch, ind, nind))
        
        print('Training on real noisy images...')
        for r in range(REAPET):
            for ind in np.random.permutation(len(train_real_fns)):
                train_real_fn = train_real_fns[ind]

                if not len(origin_real_imgs[ind]):
                    train_real_origin_fns = glob.glob(train_real_fn + '/*Reference.bmp')
                    train_real_noise_fns = glob.glob(train_real_fn + '/*Noisy.bmp')

                    origin_real_img = ReadImg(train_real_origin_fns[0])
                    origin_real_imgs[ind] = np.expand_dims(origin_real_img, axis=0)

                    for train_real_noise_fn in train_real_noise_fns:
                        noise_real_img = ReadImg(train_real_noise_fn)
                        noise_real_imgs[ind].append(np.expand_dims(noise_real_img, axis=0))

                st = time.time()
                for nind in np.random.permutation(len(noise_real_imgs[ind])):
                    H = origin_real_imgs[ind].shape[1]
                    W = origin_real_imgs[ind].shape[2]

                    ps_temp = min(H, W, PS) - 1

                    xx = np.random.randint(0, W-ps_temp)
                    yy = np.random.randint(0, H-ps_temp)
                    
                    temp_origin_img = origin_real_imgs[ind][:, yy:yy+ps_temp, xx:xx+ps_temp, :]
                    temp_noise_img = noise_real_imgs[ind][nind][:, yy:yy+ps_temp, xx:xx+ps_temp, :]
                    temp_origin_img, temp_noise_img = DataAugmentation(temp_origin_img, temp_noise_img)
                    noise_level = temp_noise_img - temp_origin_img

                    cnt += 1
                    st = time.time()

                    _, G_current, output = sess.run([G_opt_r, G_loss_r, out_image], feed_dict={in_image:temp_noise_img, gt_image:temp_origin_img, gt_noise:noise_level, lr:learning_rate})
                    output = np.clip(output, 0, 1)
                    losses.update(G_current)

                    print("%d %d Loss=%.4f Time=%.3f"%(epoch, cnt, losses.avg, time.time()-st))

                    if epoch % save_freq == 0:
                        if not os.path.isdir(result_dir + '%04d'%epoch):
                            os.makedirs(result_dir + '%04d'%epoch)

                        temp = np.concatenate((temp_origin_img[0, :, :, :], temp_noise_img[0, :, :, :], output[0, :, :, :]), axis=1)
                        scipy.misc.toimage(temp*255, high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/train_%d_%d.jpg'%(epoch, ind + len(train_syn_fns) + r * len(train_real_fns), nind))

        saver.save(sess, checkpoint_dir + 'model.ckpt')

        if not os.path.isdir(checkpoint_dir + 'epoch-' + str(epoch)):
            os.mkdir(checkpoint_dir + 'epoch-' + str(epoch))

        if os.path.isdir(checkpoint_dir + 'epoch-' + str(epoch - 1)):
            os.rmdir(checkpoint_dir + 'epoch-' + str(epoch - 1))
