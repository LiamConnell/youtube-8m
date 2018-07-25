import tensorflow as tf
import numpy as np
import os
from utils import *
from sklearn.preprocessing import MultiLabelBinarizer
import datetime
import pandas as pd
from tensorflow import flags

from GraphBuilders import * 

from apc import AveragePrecisionCalculator

import json



FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model_name", "MyModel",
    "Must define a unique name for model to be saved in")
flags.DEFINE_string(
    "graph", "simple_conv_w_res",
    "Name of graphbuilder file")
flags.DEFINE_float(
    "tstep_size", 1e-3,
    "training step size")
flags.DEFINE_integer(
    "num_files", 1000,
    "how many files should you iterate through for training")
flags.DEFINE_integer(
    "train_iters", 75,
    "how many minibatch iters per training file")
flags.DEFINE_integer(
    "batch_size", 32,
    "batch size")


directory  = '/home/ec2-user/Notebooks/Models/' + FLAGS.model_name + '/'
graph_file = eval(FLAGS.graph)
tstep_size=FLAGS.tstep_size
num_files=FLAGS.num_files
train_iters=FLAGS.train_iters
batch_size=FLAGS.batch_size

assert not os.path.exists(directory)

# BUILD GRAPH
X, Y, keep_prob, pred, loss, mloss, train = graph_file.build_graph(tstep_size=tstep_size)

# INIT
sess = tf.Session()
sess.run(tf.global_variables_initializer())






# TRAIN
one_hot = MultiLabelBinarizer(classes=[i for i in range(3862)])

tempfile = '/home/ec2-user/data/yt8m/tempfiledir/xx.tfrecord'
mirror = 'us'
partition = '2/video/train'
partition_parts = partition.split('/')
plan_url = 'http://data.yt8m.org/{}/download_plans/{}_{}.json'.format(
      partition_parts[0], partition_parts[1], partition_parts[2])
download_plan = json.loads(urllib.request.urlopen(plan_url).read())
files = [f for f in download_plan['files'].keys()]

ct = 0
start = datetime.datetime.now()
loss_plt = []
for f in files[:num_files]:
    ct+=1
    print('starting file {} of {}'.format(ct, len(files)))
    print('its been this much time: {}'.format(datetime.datetime.now()-start))
    print('{} % done'.format(int(ct/len(files)*100)))
    try:
        print('estimated time remaining {}'.format( (datetime.datetime.now()-start)/ ((ct-1)/(len(files)-(ct-1))) ))
    except:
        pass
    print(f)
    
    s=  datetime.datetime.now()
    download_url = 'http://%s.data.yt8m.org/%s/%s' % (mirror, partition, f)
    download_file(download_url, tempfile)
    
    print('download time: {}'.format(datetime.datetime.now()-s))
    
    vid_ids, labels, mean_rgb, mean_audio = get_data_from_file(tempfile)
    
    onehot_labels = one_hot.fit_transform(labels)
    b = BatchIterator(onehot_labels, mean_rgb,mean_audio)
    # TRAIN
    for i in range(train_iters):
        batch = b.next_batch(batch_size)
        if i % 74 == 0:
            loss = sess.run(mloss, feed_dict={X: b.x, Y: b.y, keep_prob:1.0})
            print('step %d, loss %g' % (i, loss))
            if i == 0:
                loss_plt.append(loss)
        sess.run(train, feed_dict={X: batch[0], Y: batch[1], keep_prob:0.5})

    print('')
    
    
# SAVE MODEL
if not os.path.exists(directory):
    os.makedirs(directory)
lossplt_dir = directory + 'loss_plt.txt'
export_dir = directory + 'Model/'
    
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

sigdef = tf.saved_model.signature_def_utils.build_signature_def(
                                        inputs= {"X": tf.saved_model.utils.build_tensor_info(X), 
                                                 "Y":tf.saved_model.utils.build_tensor_info(Y), 
                                                 "keep_prob":tf.saved_model.utils.build_tensor_info(keep_prob)},
                                        outputs= {"pred_": tf.saved_model.utils.build_tensor_info(pred)})

builder.add_meta_graph_and_variables(sess,
                                  ['tag'],
                                     signature_def_map= {"model":sigdef}        
                                    )
builder.save()

with open(lossplt_dir, 'w') as file:
    file.write(str(loss_plt))