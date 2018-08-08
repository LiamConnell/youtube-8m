import tensorflow as tf
import numpy as np
import os
from utils import *
from sklearn.preprocessing import MultiLabelBinarizer
import datetime
import pandas as pd
from tensorflow import flags

from apc import AveragePrecisionCalculator

import json

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




# INFER and WRITE CSV
datadir = '/home/ec2-user/data/yt8m/test/'
tempfile = '/home/ec2-user/data/yt8m/tempfiledir/xx.tfrecord'
outfile = 'out/'+ str(datetime.datetime.now())+'.csv'

with open(outfile, 'w') as f:
    pd.DataFrame(columns=['VideoId','LabelConfidencePairs']).to_csv(f, index=False)
    
mirror = 'us'
partition = '2/video/test'
partition_parts = partition.split('/')
plan_url = 'http://data.yt8m.org/{}/download_plans/{}_{}.json'.format(
      partition_parts[0], partition_parts[1], partition_parts[2])
download_plan = json.loads(urllib.request.urlopen(plan_url).read())
files = [f for f in download_plan['files'].keys()]

ct = 0
start = datetime.datetime.now()
for f in files:
    ct+=1
    print('starting file {} of {}'.format(ct, len(files)))
    print('its been this much time: {}'.format(datetime.datetime.now()-start))
    print('{} % done'.format(int(ct/len(files)*100)))
    try:
        print('estimated time remaining {}'.format( (datetime.datetime.now()-start)/ ((ct-1)/(len(files)-(ct-1))) ))
    except:
        pass
    print(f)
    
    download_url = 'http://%s.data.yt8m.org/%s/%s' % (mirror, partition, f)
    download_file(download_url, tempfile)
    
    vid_ids, _, mean_rgb, mean_audio = get_data_from_file(tempfile, has_labels=False)
    
    fake_labels = [[1] for _ in range(len(mean_rgb))]
    b = BatchIterator(fake_labels, mean_rgb,mean_audio)
    
    p = sess.run(tf.sigmoid(pred), feed_dict={X: b.x, keep_prob:1.0})
    
    top20 = p.argsort(axis=-1)[:,-20:][:,::-1]
    x_axis_index=np.tile(np.arange(len(p)), (top20.shape[1],1)).transpose() 
    probs = p[x_axis_index, top20]
    idx_probs = np.stack([top20, probs], axis=-1)
    idx_probs = [[(int(idx_probs[i,j,0]), idx_probs[i,j,1]) for j in range(idx_probs.shape[1])] for i in range(idx_probs.shape[0])]
    out = [str(idx_probs[i]).replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(',', '') for i in range(len(idx_probs))]
    df = pd.DataFrame(data={'VideoId':vid_ids, 'LabelConfidencePairs':out})
    
    with open(outfile, 'a') as f:
        df.to_csv(f, header=False, index=False)