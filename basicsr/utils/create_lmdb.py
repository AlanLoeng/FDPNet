# ------------------------------------------------------------------------
# Copyright (c) 2025 Bolun Liang(https://github.com/AlanLoeng) All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------
import argparse
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs
import os

def prepare_keys(folder_path, valid_extensions=None):
    if valid_extensions is None:
        valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', ".PNG"]  # 支持的图片格式
    
    img_path_list = []
    keys = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                # img_path_list.append(os.path.join(root, file))
                rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                img_path_list.append(rel_path)
                keys.append(os.path.splitext(file)[0])  # 使用文件名作为键

    print(f'Reading image path list ... Total images: {len(img_path_list)}')  # 输出找到的图像数量
    return img_path_list, keys


def create_lmdb_for_reds():
    # folder_path = './datasets/REDS/val/sharp_300'
    # lmdb_path = './datasets/REDS/val/sharp_300.lmdb'
    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    #
    # folder_path = './datasets/REDS/val/blur_300'
    # lmdb_path = './datasets/REDS/val/blur_300.lmdb'
    # img_path_list, keys = prepare_keys(folder_path, 'jpg')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/REDS/train/train_sharp'
    lmdb_path = './datasets/REDS/train/train_sharp.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/REDS/train/train_blur_jpeg'
    lmdb_path = './datasets/REDS/train/train_blur_jpeg.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def create_lmdb_for_gopro():
    folder_path = './datasets/GoPro/train/blur_crops'
    lmdb_path = './datasets/GoPro/train/blur_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/GoPro/train/sharp_crops'
    lmdb_path = './datasets/GoPro/train/sharp_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # folder_path = './datasets/GoPro/test/target'
    # lmdb_path = './datasets/GoPro/test/target.lmdb'

    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # folder_path = './datasets/GoPro/test/input'
    # lmdb_path = './datasets/GoPro/test/input.lmdb'

    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_rain13k():
    folder_path = './datasets/Rain13k/train/input'
    lmdb_path = './datasets/Rain13k/train/input.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/Rain13k/train/target'
    lmdb_path = './datasets/Rain13k/train/target.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_SIDD(compressionlevel=6):
    folder_path = './datasets/SIDD/train/input_crops'
    lmdb_path = './datasets/SIDD/train/input_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, ['.PNG', '.png'])
    
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,compress_level=compressionlevel)

    folder_path = './datasets/SIDD/train/gt_crops'
    lmdb_path = './datasets/SIDD/train/gt_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, ['.PNG', '.png'])
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,compress_level=compressionlevel)

    #for val
    '''
    
    folder_path = './datasets/SIDD/val/input_crops'
    lmdb_path = './datasets/SIDD/val/input_crops.lmdb'
    mat_path = './datasets/SIDD/ValidationNoisyBlocksSrgb.mat'
    if not osp.exists(folder_path):
        os.makedirs(folder_path)
    assert  osp.exists(mat_path)
    data = scio.loadmat(mat_path)['ValidationNoisyBlocksSrgb']
    N, B, H ,W, C = data.shape
    data = data.reshape(N*B, H, W, C)
    for i in tqdm(range(N*B)):
        cv2.imwrite(osp.join(folder_path, 'ValidationBlocksSrgb_{}.png'.format(i)), cv2.cvtColor(data[i,...], cv2.COLOR_RGB2BGR)) 
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/SIDD/val/gt_crops'
    lmdb_path = './datasets/SIDD/val/gt_crops.lmdb'
    mat_path = './datasets/SIDD/ValidationGtBlocksSrgb.mat'
    if not osp.exists(folder_path):
        os.makedirs(folder_path)
    assert  osp.exists(mat_path)
    data = scio.loadmat(mat_path)['ValidationGtBlocksSrgb']
    N, B, H ,W, C = data.shape
    data = data.reshape(N*B, H, W, C)
    for i in tqdm(range(N*B)):
        cv2.imwrite(osp.join(folder_path, 'ValidationBlocksSrgb_{}.png'.format(i)), cv2.cvtColor(data[i,...], cv2.COLOR_RGB2BGR)) 
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    '''
