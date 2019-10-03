# Design Your Idea

### Contact information
**Nguyen Dang Thanh Nguyen** (Nicky)
+84 909 200 130 - ndtnguyen130@gmail.com

![](https://i.imgur.com/rKHOnfE.jpg)


## Introduction
This Machine Learning project is a re-implementation the project in the paper **Sketch me that shoe**. It was built on Computer Vision and Recommendation System.

My purpose in this project is that I can provide for designers a tool which helps to show them many ideas for their work. The target users are shoe designers, so my application will base on their sketch of shoe and show them some kinds of shoes with same shape or feature. That will make designer's ideas come true, or it can bring them a few ideas which can motivate them in creativity at least.

## Model
### 1. Preparation
#### a. Dataset
When collect-ing the shoe photo images and sketches, we have downloaded dataset from www.eecs.qmul.ac.uk/~qian/Qian's%20Materials/ShoeV2.zip . This dataset covers **6648** sketches and **2000** photos shoes of different types including boots, high-heels, ballerinas, formal and in-formal shoes. 

![](https://i.imgur.com/Sf2iRji.png)

#### b. Environment
This project was developed on Python 2.7 and related libraries.

### 2. Training
#### One shot learning
One shot learning is the technique of learning representations from a single sample.
The problem with this task is that we might not have more sketches for each of shoe. Therefore, building and training a typical convolutional neural network will not work as it cannot learn the features required with the given amount of data. So, this is a one shot learning task where you build a similarity function which compares two images and tells you if there is a match.


#### Siamese Network
![](https://i.imgur.com/PFUUdL1.png)


In Siamese networks, we take an input image and find out the encodings of that image, then, we take the same network without performing any updates on weights or biases and input a different image and again predict it’s encodings. 
Now, we compare these two encodings to check whether there is a similarity between the two images. These two encodings act as a latent feature representation of the images. Images with the same thing have similar features/encodings. Using this, we compare and tell if the two images have the same thing or not.


#### Triplet Loss
![](https://i.imgur.com/HgXLwub.png)
The network could be trained by taking an anchor image and comparing it with both a positive sample and a negative sample. The dissimilarity between the anchor image and positive image must low and the dissimilarity between the anchor image and the negative image must be high.

![](https://i.imgur.com/4CCBItO.png)

The formula above represents the triplet loss function using which gradients are calculated. The variable “a” represents the anchor image, “p” represents a positive image and “n” represents a negative image. We know that the dissimilarity between a and p should be less than the dissimilarity between a and n,. Another variable called margin, which is a hyperparameter is added to the loss equation. Margin defines how far away the dissimilarities should be, i.e if margin = 0.2 and d(a,p) = 0.5 then d(a,n) should at least be equal to 0.7. Margin helps us distinguish the two images better.
```python
def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):
    with tf.name_scope("triplet_loss"):
        d_p_squared = square_distance(anchor_feature, positive_feature)
        d_n_squared = square_distance(anchor_feature, negative_feature)
        loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)
```

Therefore, by using this loss function we calculate the gradients and with the help of the gradients, we update the weights and biases of the siamese network. For training the network, we take an anchor image and randomly sample positive and negative images and compute its loss function and update its gradients.

**ITERATIONS for training: 20000**

### 3. Testing
**1. Store your sketch and 2000 photos of shoes into datafile .mat**
![](https://i.imgur.com/GBrEvQ1.png)

**2. Run function do_multiview_crop to generate 10 images for each photo.**
![](https://i.imgur.com/patF4xl.png)

**3. Initialize CNN network**

![](https://i.imgur.com/xxcEpWo.png)

```python
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
    
inputs = tf.placeholder(shape=[None, 225, 225, 1], dtype=tf.float32)
net = sketch_a_net_sbir(inputs)
```

**4. Load model**
```python
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
    
    
model = './model/shoes/deep_sbir/SA_SSF_HOLEF_shoes.npy'
init_ops = init_variables(model)
```
**5. Create feature vectors of sketches and photos by putting them into model**
```python
with tf.Session() as sess:
    sess.run(init_ops)
    im_feats = sess.run(net, feed_dict={inputs: im})
    sketch_feats = sess.run(net, feed_dict={inputs: sketch})
```
After this step, we will have 10 feature vectors for sketch and 20000 feature vectors for 2000 photos.

**6. Calculate average for sketch and images.**
We will calculate mean() for 10 feature vectors of sketch.
And also calculate mean() for each 10 feature vectors of photos.
After this step, we will have only 1 feature vectors for sketch and 2000 feature vectors for 2000 photos.

**7. Calculate distances between 1 feature vector of sketch to each feature vectors of 2000 photos by 'sqeuclidean' algorithm.**

![](https://i.imgur.com/nl0KZKF.png)

Finally, we will take 10 photos feature vectors which have nearest distance to sketch feature vector.
Take the indices of these vectors, now we got 10 shoes with same shape or features like the sketch.


## Application
![](https://i.imgur.com/lBnKKP8.png)

Sketch your shoe on the canvas and click on **Find now!** button.
I also provide you the 2 other buttons **Undo** and **Clear** to support your drawing.

![](https://i.imgur.com/s0aKbdi.png)

Bring your idea to universe...

![](https://i.imgur.com/wVroA6V.png)


## Conclusion
This model works well with the accuracy below:

|Dataset 	|acc.@1 	|acc.@10 	|
|-----------|:---------:|:---------:|
|Shoes 	    |52.17% 	|92.17% 	|

My target is update this project on to Python 3. with correspondent libraries.

I also try to expand this model for many categories such as: furnitures, vehicles, even characters in Art.

## References
**Paper**: https://homepages.inf.ed.ac.uk/thospeda/papers/yu2016sketchThatShoe.pdf

@inproceedings{yu2016sketch,
        title={Sketch me that shoe},
        author={Yu, Qian and Liu, Feng and Song, Yi-Zhe and Xiang, Tao and Hospedales, Timothy M and Loy, Chen Change},
        booktitle={Computer Vision and Pattern Recognition (CVPR), 2016 IEEE Conference on},
        pages={799--807},
        year={2016},
        organization={IEEE}
}

@article{yu2017sketch,
        title={Sketch-a-net: A deep neural network that beats humans},
        author={Yu, Qian and Yang, Yongxin and Liu, Feng and Song, Yi-Zhe and Xiang, Tao and Hospedales, Timothy M},
        journal={International Journal of Computer Vision},
        volume={122},
        number={3},
        pages={411--425},
        year={2017},
        publisher={Springer}
}

Links: 
https://towardsdatascience.com/siamese-network-triplet-loss-b4ca82c1aec8
