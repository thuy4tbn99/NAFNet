# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch

# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite

import os
from tqdm import tqdm
import argparse
import cv2


def main():
    opt = parse_options(is_train=False)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_path', type=str, required=True, help='The path to the input image.')
    # parser.add_argument('--output_path', type=str, required=True, help='The path to the output image.')
    # args = parser.parse_args()
    opt['dist'] = False
    model = create_model(opt)

    input_path = opt['input_path']
    output_path = opt['output_path']

    # get all image file
    arr_files_name = []
    arr_files_path = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            x = os.path.join(root, file)
            arr_files_path.append(x)
            arr_files_name.append(file)
    # print('arr_file', arr_file[0])

    pbar = tqdm(total=len(arr_files_path), unit='file')
    for idx, img_path in enumerate(arr_files_path):
        img_file = arr_files_name[idx]
        pbar.update(1)
        pbar.set_description(f"Run inference {img_file}")

        img_subfolder = (img_path.split('/')[-2:-1])[0]
        img_subfolder = os.path.join(output_path, img_subfolder)
        outp = os.path.join(img_subfolder, img_file)
        # print('outp', outp)
        
        ## 1. read image
        file_client = FileClient('disk')
        img_bytes = file_client.get(img_path, None)
        try:
            img = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("path {} not working".format(img_path))

        img = img2tensor(img, bgr2rgb=True, float32=True)

        ## 2. run inference
        model.feed_data(data={'lq': img.unsqueeze(dim=0)})
        model.test()
        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        
        if not os.path.exists(img_subfolder):
            os.makedirs(img_subfolder, exist_ok=True)
        cv2.imwrite(outp, sr_img)
        # print(f'inference {f[]} .. finished. saved to {output_path}')
    pbar.close()


if __name__ == '__main__':
    main()

