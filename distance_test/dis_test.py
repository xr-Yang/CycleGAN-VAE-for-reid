#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Xiangru Yang
@last update: 2019/10/24
@function: measure two camera image style distance
'''

import os
import cv2
import json
import time
import timeit
import shutil
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable
from scipy.spatial.distance import cdist
#from model import mcc
import random
import argparse


def Generatematfile(MODEL_PATH, camA_img_path, camB_img_path, camA_img_mat, camB_img_mat):
    extractor = pfextractor(MODEL_PATH)

    if not os.path.exists(camA_img_mat):
        os.makedirs(camA_img_mat)
    if not os.path.exists(camB_img_mat):
        os.makedirs(camB_img_mat)

#==============Generating camA features mat

    camA_img_list = os.listdir(camA_img_path)
    camA_img_list.sort()
    camA_img_list_num = len(camA_img_list)

    camA_img_mat_path = camA_img_mat + '/' + 'camA_img_mat.json'
    camA_feature_mat_path = camA_img_mat + '/' + 'camA_feature_mat.json'

    with open(camA_img_mat_path, "w") as f_camA1:
        json.dump(camA_img_list, f_camA1)
    i = 0
    camA_img_features = []
    for camA_single_img in camA_img_list:
        camA_single_img_path = camA_img_path + '/'+ camA_single_img
        camA_img = cv2.imread(camA_single_img_path)
        camA_single_img_feature = extractor.extract(camA_img)
        camA_img_features.append(camA_single_img_feature)
        i += 1
        print("camA_single_img feature: %d/%d, %s is done!" % (i, camA_img_list_num, camA_single_img))
    with open(camA_feature_mat_path, "w") as f_camA2:
        json.dump(camA_img_features, f_camA2)
    print("Generating camA matrix file completed!")

    # ==============Generating camA features mat
    camB_img_list = os.listdir(camB_img_path)
    camB_img_list.sort()
    camB_img_list_num = len(camB_img_list)

    camB_img_mat_path = camB_img_mat + '/' +  'camB_img_mat.json'
    camB_feature_mat_path = camB_img_mat + '/' + 'camB_feature_mat.json'

    with open(camB_img_mat_path, "w") as f_camB1:
        json.dump(camB_img_list, f_camB1)
    j = 0
    camB_img_features = []
    for camB_single_img in camB_img_list:
        camB_single_img_path = camB_img_path + '/' + camB_single_img
        camB_img = cv2.imread(camB_single_img_path)
        camB_single_img_feature = extractor.extract(camB_img)
        camB_img_features.append(camB_single_img_feature)
        j += 1
        print("camB_single_img feature: %d/%d, %s is done!" % (j, camB_img_list_num, camB_single_img))
    with open(camB_feature_mat_path, "w") as f_camB2:
        json.dump(camB_img_features, f_camB2)
    print("Generating camB matrix file completed!")

    return 0

def Distance_camA2B(camA_img_mat, camB_img_mat):
    camA_img_mat_path = camA_img_mat + '/' + 'camA_img_mat.json'
    camA_feature_mat_path = camA_img_mat + '/' + 'camA_feature_mat.json'

    camB_img_mat_path = camB_img_mat + '/' + 'camB_img_mat.json'
    camB_feature_mat_path = camB_img_mat + '/' + 'camB_feature_mat.json'

    with open(camA_img_mat_path,"r") as f_camA11:
        camA_img_data = json.load(f_camA11)
    with open(camA_feature_mat_path,"r") as f_camA12:
        camA_feature_data = json.load(f_camA12)

    with open(camB_img_mat_path,"r") as f_camB11:
        camB_img_data = json.load(f_camB11)
    with open(camB_feature_mat_path,"r") as f_camB12:
        camB_feature_data = json.load(f_camB12)

    all_mean = np.array([])
    for i in range(len(camA_img_data)):
        #camA_img = camA_img_data[i]
        camA_feature = camA_feature_data[i]
        camA_feature_np = np.array(camA_feature)
        camA_feature_np = np.reshape(camA_feature_np,(1, 2048))
        if camB_feature_data != []:
            distmat = (cdist(camA_feature_np, camB_feature_data, metric='cosine'))[0]
            distmat_np = np.array(distmat)
            distmat_np_mean = np.mean(distmat_np)
            print(distmat_np_mean)
            all_mean = np.insert(all_mean, 0, distmat_np_mean, axis=0)

        else:
            print('camB_feature_data is None')

    dist_average = all_mean.mean()
    #print(dist_average)
    print("the average between camA and camB is :%f"%float(dist_average))


    return 0
class pfextractor():
    def __init__(self, model_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU index
        self.model = mcc.MCC().cuda()
        self.model.load_state_dict(torch.load(model_path))

        self.transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image):
        self.model.eval()

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)
        image = image.unsqueeze_(0).float()
        image = Variable(image)

        output = self.model(image.cuda())

        f = output[0].data.cpu()
        fnorm = torch.norm(f)
        f = f.div(fnorm.expand_as(f))

        return f.tolist()


if __name__ == '__main__':

    #THRESHOLD = 0.35

    parser = argparse.ArgumentParser()

    parser.add_argument('--camA_img_path', type=str, default="../", help='camA_img_path')
    parser.add_argument('--camB_img_path', default="../", type=str, help='camB_img_path')
    parser.add_argument('--model', type=str, default="model/PED_EXT_008.pt", help='model path')
    parser.add_argument('--camA_img_mat', type=str, default="./mat/ID2_cam2_vae", help='camA_img mat path')
    parser.add_argument('--camB_img_mat', type=str, default="./mat/ID2_cam3_vae", help='camB_img mat path')
    parser.add_argument('--save_path', type=str, help='result save path')
    parser.add_argument('--generate_mat_file', action="store_true", help='generate_gallerymat_file function')
    parser.add_argument('--generate_camB_mat_file', action="store_true", help='generate_querymat_file function')
    parser.add_argument('--distance', default=False ,action="store_true", help='matching function')
    args = parser.parse_args()

    camA_img_path = args.camA_img_path
    camB_img_path = args.camB_img_path
    MODEL_PATH = args.model

    camA_img_mat = args.camA_img_mat
    camB_img_mat = args.camB_img_mat

    if args.generate_mat_file:
        print("start Generatematfile==========================")
        Generatematfile(MODEL_PATH, camA_img_path, camB_img_path, camA_img_mat, camB_img_mat)
    if args.distance:
        print("start calculate the average distance between camerasytleA and B==========================")
        Distance_camA2B(camA_img_mat, camB_img_mat)