import tensorflow as tf
import numpy as np
import os
from utils import *
from sklearn.preprocessing import MultiLabelBinarizer
import datetime
import pandas as pd
from tensorflow import flags

from apc import AveragePrecisionCalculator



FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model_name", "MyModel",
    "Must define a unique name for model to be saved in")

directory  = '/home/ec2-user/Notebooks/Models/' + FLAGS.model_name + '/'



sess = tf.Session()


# GET SAVED MODEL
# LOAD GRAPH
export_dir = directory + 'Model/'

l = tf.saved_model.loader.load(sess, 
                           ['tag'],#[tag_constants.SERVING], 
                           export_dir)

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
pred = graph.get_tensor_by_name("pred/add:0")




#EVALUATE
one_hot = MultiLabelBinarizer(classes=[i for i in range(3862)])

datadir = '/home/ec2-user/data/yt8m/validate/'
APC = AveragePrecisionCalculator()
appp = []
for file in os.listdir(datadir)[:30]:
    if file[-5:]=='.json':
        continue
    filename = datadir + file    
    vid_ids, labels, mean_rgb, mean_audio = get_data_from_file(filename)

    onehot_labels = one_hot.fit_transform(labels)
    #onehot_labels = onehot_labels[:,:869]
    
    b = BatchIterator(onehot_labels, mean_rgb,mean_audio)
    
    p = sess.run(tf.sigmoid(pred), feed_dict={X: b.x, Y: b.y, keep_prob:1.0})

    ap = 0
    for i in range(p.shape[0]):
        ap+=APC.ap(p[i], b.y[i], n=20)

    ap_avg = ap/p.shape[0]
    print(ap_avg)
    appp.append(ap_avg)
print('average: {}'.format(np.mean(ap_avg)))