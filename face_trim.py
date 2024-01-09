import cv2
import numpy as np
from tqdm import tqdm, trange

import argparse
import os
import os.path as osp
import sys
import shutil
import math


def compute_face_mask(image_parse_v3_im256, error_margin_half=(0, 0, 0)):
    im = image_parse_v3_im256
    if im.dtype != np.uint8:
        raise ValueError("Input image must be of type uint8 (0-255)")
    if len(im.shape) != 3:
        raise ValueError("Invalid image shape: {}".format(im.shape))
    face_colour = np.array([0, 0, 254])
    error_margin_half = np.array(error_margin_half)
    return cv2.inRange(im, face_colour - error_margin_half, face_colour + error_margin_half)


def main(opts):
    target_subdirs = ['image', 'agnostic-v3.2', 'image-densepose', 'image-parse-agnostic-v3.2', 'image-parse-v3', 'openpose_img']

    dsetdirs = [osp.join(opts.viton_hd_path, tt) for tt in ['train', 'test']]

    if osp.exists(opts.output_viton_hd_path):
        print("Output VITON-HD path '{}' already exists. Deleting.".format(opts.output_viton_hd_path))
        shutil.rmtree(opts.output_viton_hd_path)

    cutoff_ratio = np.clip(1.0 - opts.face_cutoff_ratio, 0.0, 1.0)
    print("Calculating cutoff point. {}% from top".format(int(100 * cutoff_ratio)))
    cutoff_y_sum = 0
    cutoff_y_n = 0
    ogdim = None
    for dsetdir in dsetdirs:
        print("Dataset dir: '{}'".format(dsetdir))
        for parsed_fn in tqdm(os.listdir(osp.join(dsetdir, 'image-parse-v3'))):
            parsed_fp = osp.join(dsetdir, 'image-parse-v3', parsed_fn)
            parsed_im = cv2.cvtColor(cv2.imread(parsed_fp), cv2.COLOR_BGR2RGB)
            ogdim = parsed_im.shape[:2][::-1]
            face_mask = compute_face_mask(parsed_im)
            yy, xx = np.nonzero(face_mask)
            if len(yy) == 0:
                print("Face not identified for: '{}'".format(parsed_fn))
            else:
                ymin, ymax = yy.min(), yy.max()
                cutoff_y = ymin + cutoff_ratio * (ymax - ymin)
                cutoff_y_sum += cutoff_y
                cutoff_y_n += 1

    cutoff_y = int(round(cutoff_y_sum / cutoff_y_n))
    print("Cutoff y: {} ({}x{} -> {}x{})".format(cutoff_y, *ogdim[:2], ogdim[0], ogdim[1] - cutoff_y))
    print("\n")

    print("Trimming images")
    for dsetdir in dsetdirs:
        print("Dataset dir: '{}'".format(dsetdir))
        for isubdir, imgs_subdirname in enumerate(target_subdirs):
            imgs_subdir = osp.join(dsetdir, imgs_subdirname)
            print("\tSubdir [{}/{}]: '{}'".format(isubdir+1, len(target_subdirs), imgs_subdirname))
            out_dir = osp.join(opts.output_viton_hd_path, osp.basename(dsetdir), imgs_subdirname)
            print("Output dir: '{}'".format(out_dir))
            os.makedirs(out_dir, exist_ok=True)
            for im_fn in tqdm(os.listdir(imgs_subdir)):
                im = cv2.imread(osp.join(imgs_subdir, im_fn))
                imc = im[cutoff_y:, :]
                cv2.imwrite(osp.join(out_dir, im_fn), imc)
    print("Done")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--viton_hd_path', '-v')
    ap.add_argument('--output_viton_hd_path', '-o')
    ap.add_argument('--face_cutoff_ratio', type=float, default=1/3.0, help="Measured from bottom of face")

    args = ap.parse_args()
    main(args)

