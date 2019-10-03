from flask import Blueprint, render_template, request, redirect, url_for
from PIL import Image, ImageOps
from io import BytesIO
from scipy.io import loadmat, savemat
import base64
import re
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt


make_sketch = Blueprint('make_sketch', __name__)
NUM_VIEWS = 10
CROPSIZE = 225

def parse_image(imgData):
    imgstr = re.search(b"base64,(.*)", imgData).group(1)
    img_decode = base64.standard_b64decode(imgstr)
    with open("./static/images/sketch.jpg", "wb") as file:
        file.write(img_decode)
    return img_decode

def read_data_file():
    # im_file = './static/data_files/im_data.mat'
    skt_file = './static/data_files/skt_data.mat'
    im_3_file = './static/data_files/im_data_3_channels.mat'
    data_skt = loadmat(skt_file)['data']
    # data_im = loadmat(im_file)['data']
    data_im_3 = loadmat(im_3_file)['data']
    return skt_file, data_skt, data_im_3

def init_variables(model_file):
    # trick to config allow_pickle=True
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    d = np.load(model_file).item()
    np.load = np_load_old
    # end trick

    pretrained_paras = d.keys()
    init_ops = []  # a list of operations

    for var in tf.global_variables():
        for w_name in pretrained_paras:
            if w_name in var.name:
                init_ops.append(var.assign(d[w_name]))

    return init_ops

def sketch_a_net_sbir(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=False):
        with slim.arg_scope([slim.conv2d], padding='VALID'):
            conv1 = slim.conv2d(inputs, 64, [15, 15], 3, scope='conv1_s1')
            conv1 = slim.max_pool2d(conv1, [3, 3], scope='pool1')
            conv2 = slim.conv2d(conv1, 128, [5, 5], scope='conv2_s1')
            conv2 = slim.max_pool2d(conv2, [3, 3], scope='pool2')
            conv3 = slim.conv2d(conv2, 256, [3, 3], padding='SAME', scope='conv3_s1')
            conv4 = slim.conv2d(conv3, 256, [3, 3], padding='SAME', scope='conv4_s1')
            conv5 = slim.conv2d(conv4, 256, [3, 3], padding='SAME', scope='conv5_s1')
            conv5 = slim.max_pool2d(conv5, [3, 3], scope='pool3')
            conv5 = slim.flatten(conv5)
            fc6 = slim.fully_connected(conv5, 512, scope='fc6_s1')
            fc7 = slim.fully_connected(fc6, 256, activation_fn=None, scope='fc7_sketch')
            fc7 = tf.nn.l2_normalize(fc7, dim=1)
    return fc7

def do_multiview_crop(fname, cropsize):
    data = loadmat(fname)['data']
    if len(data.shape) == 2: # single sketch
        data = data[np.newaxis, np.newaxis, :, :]  # nxcxhxw
    elif len(data.shape) == 3: # sketch
        n, h, w = data.shape
        data = data.reshape((n, 1, h, w))
    n, c, h, w = data.shape
    xs = [0, 0, w-cropsize, w-cropsize]
    ys = [0, h-cropsize, 0, h-cropsize]
    batch_data = np.zeros((n*10, c, cropsize, cropsize), np.single)
    y_cen = int((h - cropsize) * 0.5)
    x_cen = int((w - cropsize) * 0.5)
    for i in xrange(n):
        for (k, (x, y)) in enumerate(zip(xs, ys)):
            batch_data[i*10+k, :, :, :] = data[i, :, y:y+cropsize, x:x+cropsize]
        # center crop
        batch_data[i*10+4, :, :, :] = data[i, :, y_cen:y_cen+cropsize, x_cen:x_cen+cropsize]
        for k in xrange(5):  # flip
            batch_data[i*10+k+5, :, :, :] = batch_data[i*10+k, :, :, ::-1]
    return batch_data.transpose([0, 2, 3, 1]).astype(np.float32) - 250.42

@make_sketch.route("/results")
def results():
    return render_template("results.html")

@make_sketch.route("/upload/", methods=["POST"])
def upload_file():
    img_raw = parse_image(request.get_data())
    sketches = []
    skt = Image.open(BytesIO(img_raw))
    skt = skt.resize((256,256), Image.ANTIALIAS)
    skt = ImageOps.grayscale(skt)
    sketches.append(np.array(skt))
    sketches = np.array(sketches)
    savemat('./static/data_files/skt_data.mat', {'data':sketches})
    #####
    skt_file, data_skt, data_im_3 = read_data_file()


    inputs = tf.placeholder(shape=[None, 225, 225, 1], dtype=tf.float32)
    net = sketch_a_net_sbir(inputs)
    model = './static/models/SA_SSF_HOLEF_shoes.npy'
    init_ops = init_variables(model)

    # im = do_multiview_crop(im_file, CROPSIZE)
    sketch = do_multiview_crop(skt_file, CROPSIZE)
    with tf.Session() as sess:
        sess.run(init_ops)
        sketch_feats = sess.run(net, feed_dict={inputs: sketch})
    
    #Calculate average for sketch and images
    ave_dist_skt = sketch_feats.mean(axis=0)
    ave_dist_skt = ave_dist_skt.reshape([1, 256])
    ave_dist_img = loadmat('./static/data_files/ave_dist_img.mat')['data']
    #Calculate distance between sketch and images feature vectors
    d = ssd.cdist(ave_dist_skt,ave_dist_img)
    #Take 10 min distance
    idx = np.argpartition(d, 10)
    idx = idx.flatten()
    top_10_min = idx[:10]
    #Draw
    for i in range(len(top_10_min)):
        img = Image.fromarray(data_im_3[top_10_min[i]], 'RGB')
        img.save('./static/images/res_'+str(i)+'.png')

    return redirect(url_for('make_sketch.results'))