import tensorflow as tf

# from https://github.com/google/youtube-8m/blob/master/losses.py

def CrossEntropyLoss(predictions, labels):
    with tf.name_scope("loss_xent"):
        epsilon = 10e-6
        float_labels = tf.cast(labels, tf.float32)
        cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
            1 - float_labels) * tf.log(1 - predictions + epsilon)
        cross_entropy_loss = tf.negative(cross_entropy_loss)
        return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))
    
    
def HingeLoss(predictions, labels, b=1.0):
    with tf.name_scope("loss_hinge"):
        float_labels = tf.cast(labels, tf.float32)
        all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
        all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
        sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
        hinge_loss = tf.maximum(
            all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)
        return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))
    
    
def SoftmaxLoss(predictions, labels):
    with tf.name_scope("loss_softmax"):
        epsilon = 10e-8
        float_labels = tf.cast(labels, tf.float32)
        # l1 normalization (labels are no less than 0)
        label_rowsum = tf.maximum(
            tf.reduce_sum(float_labels, 1, keep_dims=True),
            epsilon)
        norm_float_labels = tf.div(float_labels, label_rowsum)
        softmax_outputs = tf.nn.softmax(predictions)
        softmax_loss = tf.negative(tf.reduce_sum(
          tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
    return tf.reduce_mean(softmax_loss)