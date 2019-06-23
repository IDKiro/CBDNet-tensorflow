import tensorflow as tf
import tensorflow.contrib.slim as slim

def upsample_and_sum(x1, x2, output_channels, in_channels, scope=None):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02), name=scope)
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = deconv + x2
    deconv_output.set_shape([None, None, None, output_channels])

    return deconv_output

def FCN(input):
    with tf.variable_scope('fcn'):
        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1')
        conv2 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2')
        conv3 = slim.conv2d(conv2, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3')
        conv4 = slim.conv2d(conv3, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv4')
        conv5 = slim.conv2d(conv4, 3, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv5')
        return conv5


def UNet(input):
    with tf.variable_scope('unet'):
        conv1 = slim.conv2d(input, 64, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1_1')
        conv1 = slim.conv2d(conv1, 64, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv1_2')

        pool1 = slim.avg_pool2d(conv1, [2, 2], padding='SAME')
        conv2 = slim.conv2d(pool1, 128, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2_1')
        conv2 = slim.conv2d(conv2, 128, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2_2')
        conv2 = slim.conv2d(conv2, 128, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv2_3')

        pool2 = slim.avg_pool2d(conv2, [2, 2], padding='SAME')
        conv3 = slim.conv2d(pool2, 256, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3_1')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3_2')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3_3')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3_4')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3_5')
        conv3 = slim.conv2d(conv3, 256, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv3_6')

        up4 = upsample_and_sum(conv3, conv2, 128, 256, scope='deconv4')
        conv4 = slim.conv2d(up4, 128, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv4_1')
        conv4 = slim.conv2d(up4, 128, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv4_2')
        conv4 = slim.conv2d(up4, 128, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv4_3')

        up5 = upsample_and_sum(conv4, conv1, 64, 128, scope='deconv5')
        conv5 = slim.conv2d(up5, 64, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv5_1')
        conv5 = slim.conv2d(up5, 64, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='conv5_2')

        out = slim.conv2d(conv5, 3, [1, 1], rate=1, activation_fn=None, scope='conv6')

        return out

def CBDNet(input):
    noise_level = FCN(input)

    concat_img = tf.concat([input, noise_level], 3)

    out = UNet(concat_img) + input

    return noise_level, out