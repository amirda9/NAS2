# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm

import os.path
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

import math
import sys

MODEL_DIR = './tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

def get_inception_score(images, splits=10):
    """
    Computes the Inception Score for a set of images using a pre-trained Inception v3 model.

    Args:
        images: A list of images represented as NumPy arrays.
        splits: Number of splits to use when computing the score. Default is 10.

    Returns:
        Mean and standard deviation of the Inception Score across the splits.
    """

    # print(images[0].shape)
    assert (type(images) == list)
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)

    # Load the Inception v3 model pre-trained on ImageNet
    model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg')

    # Resize images to 299x299
    resized_images = tf.image.resize(images, size=(299, 299))

    # Preprocess the images using the Inception v3 preprocessing function
    preprocessed_images = tf.keras.applications.inception_v3.preprocess_input(resized_images)

    # Compute the logits for the preprocessed images using the Inception v3 model
    logits = model(preprocessed_images)

    # Compute the softmax probabilities from the logits
    probs = tf.nn.softmax(logits)

    # Compute the mean across the batch dimension
    mean_probs = tf.reduce_mean(probs, axis=0)

    # Compute the KL divergence and Inception Score for each split
    split_scores = []
    for i in tqdm(range(splits), desc="Calculate Inception Score"):
        part = probs[i * probs.shape[0] // splits:(i + 1) * probs.shape[0] // splits, :]
        kl = part * (tf.math.log(part) - tf.math.log(mean_probs))
        kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
        split_scores.append(tf.exp(kl))

    # Compute the mean and standard deviation of the Inception Score across the splits
    mean_score, std_score = tf.math.reduce_mean(split_scores), tf.math.reduce_std(split_scores)

    return mean_score.numpy(), std_score.numpy()





# This function is called automatically.
def _init_inception():
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.io.gfile.GFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef() 
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    
    with tf.compat.v1.Session(config=config) as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                if shape._dims != []:
                    try:
                        shape = [s.value for s in shape]
                        new_shape = []
                        for j, s in enumerate(shape):
                            if s == 1 and j == 0:
                                new_shape.append(None)
                            else:
                                new_shape.append(s)
                        o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
                    except:
                        pass
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)
        sess.close()

