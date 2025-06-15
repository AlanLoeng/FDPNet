import cv2 
import numpy as np
import os
import sys
import glob
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import lmdb  # 确保已安装lmdb
from basicsr.utils.create_lmdb import make_lmdb_from_imgs, prepare_keys

def main():
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3
    opt['crop_size'] = 256
    opt['step'] = 192
    opt['thresh_size'] = 0
    
    datasets = [
        {'name': 'O-HAZE', 'path': r'F:/DeHAZE/NAFNet/datasets/O-HAZE'}
    ]
    
    for dataset in datasets:
        create_folders(dataset['path'])
        process_images(dataset['path'], opt)
        create_lmdb_for_OHaze(dataset['path'], opt['compression_level'])

def create_folders(base_path):
    os.makedirs(osp.join(base_path, 'train', 'gt'), exist_ok=True)
    os.makedirs(osp.join(base_path, 'train', 'hazy'), exist_ok=True)
    os.makedirs(osp.join(base_path, 'val', 'gt'), exist_ok=True)
    os.makedirs(osp.join(base_path, 'val', 'hazy'), exist_ok=True)


def process_images(base_path, opt):
    for keyword in ['hazy', 'GT']:
        input_folder = osp.join(base_path, keyword)
        img_list = sorted(glob.glob(osp.join(input_folder, '*.*')))
        
        # Select the last 5 images for validation
        val_images = img_list[-5:]
        train_images = img_list[:-5]
        
        opt['input_folder'] = input_folder
        opt['save_folder'] = osp.join(base_path, 'train', keyword)
        extract_subimages(opt, train_images)

        opt['save_folder'] = osp.join(base_path, 'val', keyword)
        extract_subimages(opt, val_images)

def extract_subimages(opt, img_list):
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')

def worker(path, opt):
    crop_size = opt['crop_size']
    img_name, extension = osp.splitext(osp.basename(path))
    
    # 提取序号
    sequence_number = img_name.split('_')[0]  
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print(f"Error loading image {path}")
        return f"Failed to load {path}"
    
    h, w = img.shape[:2]
    h_space = np.arange(0, h - crop_size + 1, opt['step'])
    w_space = np.arange(0, w - crop_size + 1, opt['step'])
    
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size]
            # 保存时确保扩展名
            cv2.imwrite(osp.join(opt['save_folder'], f'{sequence_number}_s{index:03d}{extension}'), cropped_img, [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    return f'Processing {img_name} ...'


def create_lmdb_for_OHaze(base_path, compress_level=1):
    for phase in ['train', 'val']:
        for keyword in ['hazy', 'GT']:
            folder_path = osp.join(base_path, phase, keyword)  
            lmdb_path = osp.join(base_path, phase, keyword + '.lmdb')  
            
            print(f'Reading image path list from {folder_path} ...')
            img_path_list, keys = prepare_keys(folder_path, None)  
            
            print(f'Image path list: {img_path_list}')
            
            if not img_path_list:
                print(f'No images found in {folder_path}. Skipping LMDB creation.')
                continue
            
            print(f'Create lmdb for {folder_path}, save to {lmdb_path}...')
            print(img_path_list)
            make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys, compress_level=compress_level)

if __name__ == '__main__':
    main()
