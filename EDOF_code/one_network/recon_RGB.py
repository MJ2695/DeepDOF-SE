import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import glob
import time
# import scipy.misc
from PIL import Image
import Network_RGB
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only uses GPU 1



#### Where the trained models are saved
result_dir = "where_your_trained_weights_are_stored"
result_best_dir = "where_your_trained_weights_are_stored/best_model/"

#### where the images are saved
image_path = "where_your_image_to_be_reconstructed_is_stored"
save_dir_RGB = "where_you_want_to_store_the_reconstructed_images"

############ Put data in batches #############
##  put in batch and shuffle
##  cast to float32
##  call data_augment for image preprocess
## @param{TFRECORD_PATH}: path to the data
## @param{batchsize}: currently 21 for the 21 PSFs
##############################################
def read2batch(PM_img, r, c, r_size, c_size):
    offset_y = r * 16 * r_size
    offset_x = c * 16 * c_size

    # crop the FOV
    PM_img_curr = PM_img[offset_y: offset_y + 16 * r_size, offset_x: offset_x + 16 * c_size, :]
    PM_img_curr = tf.expand_dims(PM_img_curr, axis=0)
    PM = tf.tile(PM_img_curr, [batch_size, 1, 1,1]) #batch_size x r_size x c_size x 3

    return PM


def unflip(RGB_hat_phasemask_img, r, c, r_size, c_size, PM_img_recon):
    offset_y = r * r_size
    offset_x = c * c_size

    # flip the image
    PM_hat1 = RGB_hat_phasemask_img[0, :, :, :]
    PM_img_recon[offset_y:offset_y + 16 * r_size, offset_x:offset_x + 16 * c_size, :] = PM_hat1
    return PM_img_recon


####################### system ##########################
## @param{PSFs}: the PSFs
## @param{RGB_batch_float}: patches
## @param{phase_BN}: batch normalization, True only during training
########################################################
def system(pm_blur, phase_BN=False):
    with tf.variable_scope("system", reuse=tf.AUTO_REUSE):
        RGB_hat_pm = Network_RGB.UNet(pm_blur, phase_BN)
        return RGB_hat_pm


######################  RMS cost #############################
## @param{GT}: ground truth
## @param{hat}: reconstruction
##############################################################
def cost_rms(GT, hat):
    cost = tf.sqrt(tf.reduce_mean(tf.reduce_mean((tf.square(GT - hat)), 1), 1))
    return cost


def npy_to_images(img, save_dir_RGB, save_dir_R, save_dir_B, save_name):
    img = np.uint8((img/np.amax(img))*255)
    img_save = Image.fromarray(img, 'RGB')
    img_save.save(save_dir_RGB+save_name)


batch_size = 1  # only 1 image patch at a time
r_size = 114
c_size = 171


def load_img(img_name):
    # load image
    pm_path = str(Path(img_name))

    # modify the input
    PM_img = cv2.imread(pm_path)
    PM_img = cv2.cvtColor(PM_img, cv2.COLOR_BGR2RGB)
    PM_img = 0.9 * PM_img / 255
    PM_img = PM_img[0:3648, 0:5472, :]  # size of image from the camera
    PM_img = np.pad(PM_img, ((0, 3648-PM_img.shape[0]),(0,5472-PM_img.shape[1]), (0,0)), mode='constant')
    print(PM_img.shape)
    PM_img = np.rot90(PM_img, k=2)
    return PM_img


# allocate empty matrix
PM_img = tf.placeholder(dtype=tf.float32, shape=[3648, 5472, 3])

# break into patch because original image is too large
PM1 = read2batch(PM_img, 0, 0, r_size, c_size)
PM2 = read2batch(PM_img, 0, 1, r_size, c_size)
PM3 = read2batch(PM_img, 1, 0, r_size, c_size)
PM4 = read2batch(PM_img, 1, 1, r_size, c_size)

# feed into system
RGB_hat_phasemask1 = system(PM1)
RGB_hat_phasemask2 = system(PM2)
RGB_hat_phasemask3 = system(PM3)
RGB_hat_phasemask4 = system(PM4)

saver_best = tf.train.Saver()

with tf.Session() as sess:
    # threading for parallel
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # restore saver
    # saver_best.restore(sess, result_best_dir + "model.ckpt")
    saver_best.restore(sess, result_dir + "model.ckpt-99990")

    for filename in glob.glob(image_path+'/*.png'):
        print(filename)
        curr_img = load_img(filename)
        PM_img_recon = np.zeros((3648, 5472, 3))

        t = time.time()
        RGB_hat_phasemask_img1 = sess.run(RGB_hat_phasemask1, feed_dict={PM_img: curr_img})
        RGB_hat_phasemask_img2 = sess.run(RGB_hat_phasemask2, feed_dict={PM_img: curr_img})
        RGB_hat_phasemask_img3 = sess.run(RGB_hat_phasemask3, feed_dict={PM_img: curr_img})
        RGB_hat_phasemask_img4 = sess.run(RGB_hat_phasemask4, feed_dict={PM_img: curr_img})
        elapsed = time.time() - t
        print(elapsed)

        # put reconstruction into large empty matrix
        PM_img_recon[0:16 * r_size, 0:16 * c_size, :] = RGB_hat_phasemask_img1[0, :, :, :]
        PM_img_recon[0:16 * r_size, 16 * c_size:2 * 16 * c_size, :] = RGB_hat_phasemask_img2[0, :, :, :]
        PM_img_recon[16 * r_size:2 * 16 * r_size, 0:16 * c_size, :] = RGB_hat_phasemask_img3[0, :, :, :]
        PM_img_recon[16 * r_size:2 * 16 * r_size, 16 * c_size:2 * 16 * c_size, :] = RGB_hat_phasemask_img4[0, :, :, :]


        PM_img_recon5 = np.rot90(PM_img_recon, k=2)
        # print(image_path)
        save_name = filename.replace(image_path, "")
        # print(save_name)
        npy_to_images(PM_img_recon5, save_dir_RGB, save_dir_R, save_dir_B, save_name)

    coord.request_stop()
    coord.join(threads)



