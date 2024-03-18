#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-12-12 13:58
# @Author  : Siyuan Chen
# @Site    : 
# @File    : inference.py
# @Software: PyCharm
from PIL import Image
import os,sys
from tqdm import tqdm
import cv2
from BRChromNet import BRChromNet
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets, models
import torch
from ensembled_discriminator import get_resnet50,get_efficientnetV2,get_convnext

from banding_pattern_extraction.scripts.banding_pattern_extraction import get_banding_pattern
from banding_pattern_extraction.scripts.lib.visualisation_utils import binary_vector_to_bp_image,path_to_mat



def identify_abnormal_banding(ab_mask, bp_vector):
    min_val = np.min(ab_mask)
    max_val = np.max(ab_mask)
    normalized = (ab_mask - min_val) * 255 / (max_val - min_val)
    ab_bp_vector = bp_vector.copy()
    normalized = normalized.astype(np.uint8)
    mean_value = np.mean(normalized[:, 25])



    # plt.imshow(normalized)
    # plt.show()
    # plt.savefig('debug.png')
    # input()
    _, thresholded_image = cv2.threshold(normalized, mean_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # debug_img = normalized.copy()
    # cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Draw the rectangle


    max_counter_len = 0
    c_id = 0
    for i in range(len(contours)):
        if len(contours[i]) > max_counter_len:
            c_id = i
            max_counter_len = len(contours[i])

    x, y, w, h = cv2.boundingRect(contours[c_id])


    # search_string = ''
    # for c in bp_vector:
    #     search_string += str(c)
    # max_bp = longest_connected_strings_with_char(search_string)
    # max_len_bp = len(max_bp)
    #
    # if h > max_len_bp:
    for i_height in range(y, y + h):
        ab_bp_vector[i_height] = 2
    return ab_bp_vector

def resize_and_pad(image_matrix, desired_size):
    # Calculate aspect ratios
    original_height, original_width = image_matrix.shape[:2]
    desired_width, desired_height = desired_size

    # Calculate aspect ratios for resizing
    aspect_ratio = min(desired_width / original_width, desired_height / original_height)
    new_width = int(original_width * aspect_ratio)
    new_height = int(original_height * aspect_ratio)

    # Resize the image using OpenCV
    resized_image = cv2.resize(image_matrix, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a white canvas of the desired size

    padded_image = np.full((desired_height, desired_width), 255, dtype=np.uint8)

    # Calculate the position to paste the resized image onto the padded image
    paste_x = (desired_width - new_width) // 2
    paste_y = (desired_height - new_height) // 2

    # Paste the resized image onto the padded image
    padded_image[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = resized_image

    return padded_image

def get_band_img(inpt_img):
    results = get_banding_pattern(inpt_img)
    step_vector = 1
    binarized_banding_pattern = results['binarized_banding_pattern']
    bp_img = binary_vector_to_bp_image(binarized_banding_pattern)
    return cv2.rotate(bp_img, cv2.ROTATE_90_CLOCKWISE), binarized_banding_pattern


def chromosome_mask_segmentation(img, bp_ab):
    original_size = img.shape

    chromosome_segmentation = np.zeros_like(img)
    counter = np.zeros_like(img)

    results = get_banding_pattern(img)
    bp = results['binarized_banding_pattern']
    bp_points = results['banding_points']
    blob = results['blobs']

    for i in range(len(bp_ab)):
        banding_pattern_line = np.array(bp_points[i])

        if bp_ab[i] != 2:
            continue

        try:
            r = banding_pattern_line[:, 1]
            c = banding_pattern_line[:, 0]
            chromosome_segmentation[c, r] += 0
            counter[c, r] += 1
            counter[c - 0, r] += 1
            counter[c + 1, r] += 1
            counter[c, r - 1] += 1
            counter[c, r + 1] += 1
        except:  # Empty line, simply ignore
            pass

    divison_mask = np.where(counter > 0)

    chromosome_segmentation[divison_mask] = chromosome_segmentation[divison_mask] / counter[divison_mask]
    chromosome_segmentation = np.round(chromosome_segmentation)
    chromosome_segmentation = chromosome_segmentation * 127.5  # white bands should be gray

    final_segmentation = np.ones_like(img) * 255  # background should be white
    final_segmentation[divison_mask] = chromosome_segmentation[divison_mask]

    return final_segmentation

def get_prediction_mask(band_img, model):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([235.7210, 235.7210, 235.7210], [44.3981, 44.3981, 44.3981])  # imagenet
    ])
    with torch.no_grad():
        inpt_img = trans(band_img)
        inpt_img = inpt_img.unsqueeze(0)
        inputs = inpt_img.to(device)
        prediction_test = model(inputs)
        prediction_test = torch.sigmoid(prediction_test)
        prediction_test = prediction_test.data.cpu().numpy()
        mask_pred = prediction_test[0,0,:,:]
    return mask_pred


def abnormal_classification(inpt_img):
    # Load Classification Model
    model_1 = get_resnet50(OUTPUT_DIM=2)
    model_1.load_state_dict(
        torch.load('models/{}/resnet50_best.pt'.format(
            abnormal_type), ))
    model_1 = model_1.to(device)
    model_1.eval()

    model_2 = get_efficientnetV2(OUTPUT_DIM=2)
    model_2.load_state_dict(
        torch.load('models/{}/efficientnetV2_best.pt'.format(
            abnormal_type), ))
    model_2 = model_2.to(device)
    model_2.eval()

    model_3 = get_efficientnetV2(OUTPUT_DIM=2)
    model_3.load_state_dict(
        torch.load('models/{}/efficientnetV2_best.pt'.format(
            abnormal_type), ))
    model_3 = model_3.to(device)
    model_3.eval()

    pretrained_size = 256
    pretrained_means = [0.9249, 0.9251, 0.9250]
    pretrained_stds = [0.1706, 0.1706, 0.1706]
    test_transforms = transforms.Compose([
        transforms.Resize(pretrained_size),
        transforms.CenterCrop(pretrained_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                             std=pretrained_stds)
    ])
    PIL_image = Image.fromarray(inpt_img)
    inpt_img = test_transforms(PIL_image)
    inpt_img = torch.unsqueeze(inpt_img, 0)
    with torch.no_grad():
        x = inpt_img.to(device)
        out_1, _ = model_1(x)
        prob_1 = F.softmax(out_1, dim=-1)
        pred_1 = prob_1.argmax(1, keepdim=True).detach().cpu().numpy()
        out_2 = model_2(x)
        prob_2 = F.softmax(out_2, dim=-1)
        pred_2 = prob_2.argmax(1, keepdim=True).detach().cpu().numpy()

        out_3 = model_2(x)
        prob_3 = F.softmax(out_3, dim=-1)
        pred_3 = prob_3.argmax(1, keepdim=True).detach().cpu().numpy()

    final_pred = int((pred_1[0][0]+ pred_2[0][0]+ pred_3[0][0]) / 3)
    return final_pred

if __name__ == '__main__':
    # abnormal_type = '1qh+'
    # chrom_type = 'chr1'

    # abnormal_type = '9qh+'
    # chrom_type = 'chr9'

    abnormal_type = 'inv9_p12q13'
    chrom_type = 'chr9'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BRChromNet()
    model = model.to(device)
    model.load_state_dict(torch.load('weight/{}_best.pt'.format(abnormal_type)))
    model.eval()



    for normal_type in ['abnormal', 'normal']:

        save_root = 'inference-output/{}'.format(abnormal_type)
        if not os.path.exists(os.path.dirname(save_root)):
            os.mkdir(os.path.dirname(save_root))
        if not os.path.exists(save_root):
            os.mkdir(save_root)

        go_through_dir = 'data/{}/{}/'.format(abnormal_type,normal_type)

        test_abnormal_imgs = []
        for img_id in os.listdir(go_through_dir):
            test_abnormal_imgs.append(img_id)

        for test_img in tqdm(test_abnormal_imgs):

            if '.npy' in test_img:
                continue

            original_img = cv2.imread(os.path.join(go_through_dir,test_img), 0)
            band_img, bp_vector = get_band_img(original_img)

            padded_band = resize_and_pad(np.array(band_img, dtype=np.uint8), (256, 256))
            channel_image = cv2.cvtColor(padded_band, cv2.COLOR_RGB2BGR)

            # First conduct binary classification
            pred_label = abnormal_classification(channel_image)

            if pred_label == 1:
                mask_pred = get_prediction_mask(channel_image, model)[:, 101:151]
                # Resize the image using OpenCV
                band_mask = cv2.resize(mask_pred, (band_img.shape[1], band_img.shape[0]), interpolation=cv2.INTER_AREA)
                ab_bp_vector = identify_abnormal_banding(band_mask, bp_vector)
                mask_prediction = chromosome_mask_segmentation(original_img, ab_bp_vector)
                mask_prediction = (mask_prediction- 255)*255
            else:
                mask_prediction = np.zeros_like(original_img)

            save_path = os.path.join(save_root, normal_type)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            rows, cols = np.where(mask_prediction == 255)
            # Combine the row and column indices
            coordinates = list(zip(rows, cols))

            save_file = os.path.join(save_path,'{}.npy'.format(test_img.strip('.png')))
            np.save(save_file, mask_prediction)

