import tensorflow as tf
import numpy as np
import glob
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.misc
from scipy.ndimage.interpolation import rotate
from pathlib import Path
import Network_c1
import Network_c2
import os
import cv2
import time
from PIL import Image


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # only uses GPU 1


#### Where the trained models are saved
result_dir = "where_your_trained_weights_are_stored"
result_best_dir = "where_your_trained_weights_are_stored/best_model/"

#### where the images are saved
image_path = "where_your_image_to_be_reconstructed_is_stored"
save_dir = "where_you_want_to_store_the_reconstructed_images"

############ convert image into batch  #############
##  crop image
##  put RGB image in batch: [batchsize, N, N, 2]
##  only red and blue channels are used
##  during testing, batchsize=1
##  @param{PM_img}: read image in RGB format
##  @param{r,c}: offset value in row and column
##  @param{r_size,c_size}: number of rows and columns
####################################################
def read2batch(PM_img, r, c, r_size, c_size):
    offset_y = r
    offset_x = c

    # crop the FOV
    PM_img = PM_img[offset_y: offset_y + 16 * r_size, offset_x: offset_x + 16 * c_size, :]
    # put in batch: [N,N,3] to [batchsize=1,N,N,2]
    PM_img_c1 = PM_img[:,:,0:1]
    PM_img_c2 = PM_img[:,:,2:3]
    
    PM_img_curr = tf.concat([PM_img_c1, PM_img_c2], axis=2) #remove green component of RGB; convert to [N,N,2]
    PM_img_curr = tf.expand_dims(PM_img_curr, axis=0)   #add batchsize=1;convert to [1, N, N, 2]
    # print('PM_img_curr shape', PM_img_curr.get_shape().as_list())
    return PM_img_curr


####################### system ###########################################
## @param{pm_blur}: DeepDOF MUSE images, in [batchsize, N, N, 2] format
## @param{phase_BN}: batch normalization, True only during training
##########################################################################
def system(pm_blur, phase_BN):
    with tf.variable_scope("system", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("network_c1"):
            RGB_hat_c1 = Network_c1.UNet(pm_blur[:, :, :, 0:1], phase_BN)

        with tf.variable_scope("network_c2"):
            RGB_hat_c2 = Network_c2.UNet(pm_blur[:, :, :, 1:2], phase_BN)

        RGB_hat_pm = tf.concat([RGB_hat_c1, RGB_hat_c2], axis=3)     #concat to generate [batchsize, N, N, 2]

        return RGB_hat_pm


def npy_to_images(img_cur, save_name):
    img_min = np.amin(np.ndarray.flatten(img_cur))
    img_max = np.amax(np.ndarray.flatten(img_cur))
    img_cur = (img_cur * 255).astype(np.uint8)
    img_cur = Image.fromarray(img_cur,mode='RGB')
    img_cur.save(save_name)


batch_size = 1  # only 1 image patch at a time
# size of FOV (3008x3008 pixels)
r_size = 188
c_size = 188

################ load RGB image ##################
## load RGB image
## cast to float
## normalize and rotate
## used a 0.5 factor to improve robustness against noise and artifacts
## @ param{img_name}: image name
##################################################
def load_img(img_name):
    # load image
    pm_path = str(Path(img_name))

    # modify the input
    PM_img = cv2.imread(pm_path)    #cv2.imread() loads images as BGR
    PM_img = PM_img[:,:,::-1]       #BGR -> RGB
    # PM_img = PM_img[174:174+16*c_size, 174:174+16*r_size]  # size of image from the camera; smaller regions
    PM_img = 0.5*PM_img.astype(np.float32) / 255 # use system batch normalization instead of direct normalization; importantly, use float
    PM_img = np.rot90(PM_img, k=2, axes=(0, 1)) #rot180 is used bc convolution is correlation in python; define axe to avoid rot in color

    return PM_img


############################## build testing pipeline #####################################
# allocate empty matrix for a single RGB image
PM_img = tf.placeholder(dtype=tf.float32, shape=[3648,5472,3])
PM = read2batch(PM_img, 3648-3008-250, 5472-3008-950, r_size, c_size) #note the rotation in load_img...

# feed into system
RGB_hat_phasemask = system(PM, phase_BN=False)


############################## start reconstruction #######################################
saver_best = tf.train.Saver()

with tf.Session() as sess:
    # threading for parallel
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # restore saver
    # saver_best.restore(sess, result_best_dir + "model.ckpt")
    saver_best.restore(sess, result_dir + "model.ckpt-399990")

    #create folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in glob.glob(image_path+'/*.png'):
        print(filename)
        curr_img = load_img(filename)

        PM_img_recon = np.zeros((16*r_size, 16*c_size,3))
        RGB_hat_phasemask_img = sess.run(RGB_hat_phasemask, feed_dict={PM_img: curr_img})

        # put reconstruction into large empty matrix
        PM_img_recon[:,:,0] = RGB_hat_phasemask_img[0, :, :, 0]
        PM_img_recon[:,:,2] = RGB_hat_phasemask_img[0, :, :, 1]

        PM_img_recon = np.rot90(PM_img_recon, k=2)
        save_name = filename.replace(image_path, "").replace(".tif", ".png")
        PM_img_recon = np.clip(2*PM_img_recon, 0, 1)
        npy_to_images(PM_img_recon, save_dir + save_name)
        # npy_to_images(PM_img_recon, save_dir + '206re.tif')


    coord.request_stop()
    coord.join(threads)