from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt


im_3_file = './static/data_files/im_data_3_channels.mat'
data_im_3 = loadmat(im_3_file)['data']

top_10_file = './static/data_files/top_10.mat'
top_10_min = loadmat(top_10_file)['data']

print(top_10_min)