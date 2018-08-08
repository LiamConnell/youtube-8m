import tensorflow as tf
from .utils import *
from .loss_fns import *

def build_graph(tstep_size, loss_fn='CrossEntropyLoss'):
    xdim = 1024+128
    ydim = 3862
    X = tf.placeholder(tf.float32, shape=[None, xdim], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, ydim], name='Y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

#     feature_dim = len(X.get_shape()) - 1
#     X_norm = tf.nn.l2_normalize(X, feature_dim)
    
    x_img, x_audio = tf.split(X, [1024, 128], axis=-1)
    
    feature_dim_img = len(x_img.get_shape()) - 1
    x_img = tf.nn.l2_normalize(x_img, feature_dim_img)
    feature_dim_audio = len(x_audio.get_shape()) - 1
    x_audio = tf.nn.l2_normalize(x_audio, feature_dim_audio)

    img_reshape = tf.reshape(x_img, [-1, 32, 32, 1])

    h1 = lrelu(conv2d(img_reshape, 1, 16, 'conv_1')) # 32x32x1 -> 16x16x16
    h2 = lrelu(conv2d(h1, 16, 32, 'conv_2'))# 16x16x16 -> 8x8x32
    h2_flat = tf.reshape(h2,[-1, 8*8*32])
    h2_flat = slim.batch_norm(h2_flat)
    h1_flat = tf.reshape(h1,[-1, 16*16*16])
    h1_flat = slim.batch_norm(h1_flat)
    
    h_aud = lrelu(dense(x_audio, 128, 100))

    concat = tf.concat([h1_flat, h2_flat, x_audio, h_aud], 1)
    concat = slim.batch_norm(concat)
                  
    h = lrelu(dense(concat, 16*16*16 + 8*8*32 + 128 + 100, 200, scope='h'))
    h = slim.batch_norm(h)
    #h = slim.dropout(h, keep_prob=keep_prob)
                  
    pred = dense(h, 200, ydim, scope = 'pred')

    sig = tf.sigmoid(pred)
    
    
    
    if loss_fn == 'CrossEntropyLoss':
        loss = CrossEntropyLoss(sig, Y)
    if loss_fn == 'HingeLoss':
        loss = HingeLoss(sig, Y)
    if loss_fn == 'SoftmaxLoss':
        loss = SoftmaxLoss(sig, Y)
    
    mloss = loss
   

    train = tf.train.AdamOptimizer(tstep_size).minimize(loss)
    
    return X, Y, keep_prob, pred, loss, mloss, train