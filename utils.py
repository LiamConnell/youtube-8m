import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#### PLOTTING
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[:16]):
        sample = np.array(sample)
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


#### NN SOUP INGREDIENTS
class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

# standard convolution layer
def conv2d(x, inputFeatures, outputFeatures, name):
    with tf.variable_scope(name):
        w = tf.get_variable("w",[5,5,inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputFeatures], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, w, strides=[1,2,2,1], padding="SAME") + b
        return conv
    
# standard convolution layer with pooling (much slower!)
def conv2d_with_pooling(x, inputFeatures, outputFeatures, name):
    with tf.variable_scope(name):
        w = tf.get_variable("w",[5,5,inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputFeatures], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding="SAME") + b
        conv = lrelu(conv)
        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
        return pool

def conv_transpose(x, outputShape, name):
    with tf.variable_scope(name):
        # h, w, out, in
        w = tf.get_variable("w",[5,5, outputShape[-1], x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b",[outputShape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1,2,2,1])
        return convt

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

# leaky reLu unit
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# fully-conected layer
def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias
        
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
# def xavier_init(fan_in, fan_out, constant=1): 
#     """ Xavier initialization of network weights"""
#     # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
#     low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
#     high = constant*np.sqrt(6.0/(fan_in + fan_out))
#     return tf.random_uniform((fan_in, fan_out), 
#                              minval=low, maxval=high, 
#                              dtype=tf.float32)





#### BATCH ITERATOR
class train_iterator():
    def __init__(self, data):
        self.x, self.y = data
        self.pointer = 0
    def next_batch(self, N):
        x = self.x[self.pointer: self.pointer+N]
        y = self.y[self.pointer: self.pointer+N]
        self.pointer += N
        if self.pointer >= self.x.shape[0]:
            self.pointer = 0
        return x, y
    
    
    
    
    
    
    
    
    
def download_file(source_url, destination_path):
    """Downloads `source_url` onto `destionation_path`."""
    def _progress(count, block_size, total_size):
        sys.stderr.write('\r>> Downloading %s %.1f%%' % (
            source_url, float(count * block_size) / float(total_size) * 100.0))
        #sys.stderr.flush()
    urllib.request.urlretrieve(source_url, destination_path)#, _progress)
    statinfo = os.stat(destination_path)
    print('Succesfully downloaded', destination_path, statinfo.st_size, 'bytes.')
    return destination_path


def get_data_from_file(filename, has_labels=True):
    vid_ids, labels, mean_rgb, mean_audio = [], [], [], []

    for example in tf.python_io.tf_record_iterator(filename):
        tf_example = tf.train.Example.FromString(example)

        vid_ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
        labels.append(tf_example.features.feature['labels'].int64_list.value)
        mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
        mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
    return vid_ids, labels, mean_rgb, mean_audio

class BatchIterator:
    def __init__(self, labels, mean_rgb, mean_audio):
        r = np.array(mean_rgb)
        a = np.array(mean_audio)
        y = np.array(labels)
        x = np.concatenate([r,a], axis=1)
        self.x = x
        self.y = y
        assert self.x.shape[0] == self.y.shape[0], 'X, Y not same shape'
        
        self.pointer = 0
        self.data_len = self.x.shape[0]
        self.max_labels = max([max(x) for x in labels])
    def next_batch(self, n):
        if self.pointer+n>self.data_len:
            self.pointer=0
        y_mb = self.y[self.pointer:self.pointer+n]
        x_mb = self.x[self.pointer:self.pointer+n]
        self.pointer += n
        return x_mb, y_mb
    

    

        
        