# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

import os
import glob

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from skimage.transform import resize
import imageio

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *


# only keep warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder', type=str,
                    help='type of encoder, vgg or resnet50', default='resnet50')
parser.add_argument('--image_dir', type=str, help='path to the image', required=True)
parser.add_argument('--output_dir', type=str, help='path to store results', required=True)
parser.add_argument('--checkpoint_path', type=str,
                    help='path to a specific checkpoint to load',
                    default='pretrained/model_city2kitti_resnet')
parser.add_argument('--input_height', type=int, help='input height', default=256)
parser.add_argument('--input_width', type=int, help='input width', default=512)

args = parser.parse_args()


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def plot_result(input_image, disp_pp, output_png):
    """Load in the pano and the depth estimate, and plot them together"""
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    im1 = ax1.imshow(input_image)
    im2 = ax2.imshow(disp_pp, cmap='plasma')

    fig.savefig(output_png[:-3] + '_combined.png')
    plt.close(fig)


def load_image(image_path):
    input_image = imageio.imread(image_path, pilmode="RGB")
    if input_image.shape[2] == 4:
        # drop alpha channel, can't deal with that here and not necessary
        input_image = input_image[:, :, :3]

    original_shape = input_image.shape
    input_image = resize(input_image,
                         (args.input_height, args.input_width),
                         anti_aliasing=True)
    input_image = input_image.astype(np.float32)
    input_image_lr = np.stack((input_image, np.fliplr(input_image)), 0)
    return input_image, input_image_lr, original_shape


def process_directory(params):
    """Adapted from `monodepth_simple.py` - process 1 dir instead of 1 file"""

    # Load model
    left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
    model = MonodepthModel(params, "test", left, None)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    output_directory = os.path.dirname(args.output_dir)

    # Process all images
    images_to_process = glob.glob(args.image_dir + '*.png')
    images_to_process.extend(glob.glob(args.image_dir + '*.jpg'))

    for image_path in images_to_process:
        # Load image
        input_image, input_image_lr, original_shape = load_image(image_path)

        # Run model on image
        disp = sess.run(model.disp_left_est[0], feed_dict={left: input_image_lr})
        disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

        output_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save array of disparity values
        np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)),
                disp_pp)
        # Save result png's
        #disp_to_img = resize(disp_pp.squeeze(), original_shape[:2])
        output_png = os.path.join(output_directory, "{}_disp.png".format(output_name))
        plt.imsave(output_png, disp_pp, cmap='plasma')
        #plot_result(input_image, disp_pp, output_png)

    print('done!')


def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    process_directory(params)


if __name__ == '__main__':
    tf.app.run()
    # python monodepth_simple_ondirectory.py --image_dir testdata --output_dir testdata
    # OR
    #python monodepth_batch_2.py --image_dir ../thesisStef/001_data/00_classification/test/ --output_dir ../thesisStef/001_data/00_classification/test_out/
    # python monodepth_batch_2.py --image_dir ../thesisStef/001_data/00_classification/Vancouver_GSV_Tree_Pack1_40K/ --output_dir ../thesisStef/001_data/00_classification/Vancouver_pack1_disp/
    # python monodepth_simple_ondirectory.py --image_dir /home/stef/mthesis/data/dataset/train/ --output_dir /home/stef/mthesis/data/monodepth/dataset/train/

