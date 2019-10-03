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
When collect-ing the shoe photo images and sketches, we have downloaded dataset from www.eecs.qmul.ac.uk/~qian/Qian's%20Materials/ShoeV2.zip . This dataset covers **2000** sketches and **6648** photos shoes of different types including boots, high-heels, ballerinas, formal and in-formal shoes. 

![](https://i.imgur.com/Sf2iRji.png)

#### b. Environment
This project was developed on Python 2.7 and related libraries.

### 2. Training
https://towardsdatascience.com/siamese-network-triplet-loss-b4ca82c1aec8

### 3. Testing
Store sketches and photos into datafile .mat
do_multiview_crop
initialize cnn network
```python
inputs = tf.placeholder(shape=[None, 225, 225, 1], dtype=tf.float32)
net = sketch_a_net_sbir(inputs)
```
load model from file npy
```python
model = './model/shoes/deep_sbir/SA_SSF_HOLEF_shoes.npy'
init_ops = init_variables(model)
```
create feature vectors of sketches and photos by putting them into model
```python
with tf.Session() as sess:
    sess.run(init_ops)
    im_feats = sess.run(net, feed_dict={inputs: im})
    sketch_feats = sess.run(net, feed_dict={inputs: sketch})
```

Calculate average for sketch and images.

compute_view_specific_distance 'sqeuclidean'.

Take 10 photos feature vectors which have nearest distance to sketch feature vector.

## Application
![](https://i.imgur.com/lBnKKP8.png)

Sketch your shoe on the canvas and click on **Find now!** button.
I also provide you the 2 other buttons **Undo** and **Clear** to support your drawing.

![](https://i.imgur.com/s0aKbdi.png)

Bring your idea to universe...

![](https://i.imgur.com/wVroA6V.png)


## Conclusion
This model works well with the accuracy below:
Dataset 	acc.@1 	acc.@10 	%corr.
Shoes 	52.17% 	92.17% 	72.29%

My target is update this project on to Python 3. with correspondent libraries.

I also try to expand this model for many categories such as: furnitures, vehicles, even characters in Art.
