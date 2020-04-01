#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: xr Yang
@last update: 2019/09/02
@function: re-id search for annotator
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
from model import mcc
import random
import argparse


def Generatematfile(MODEL_PATH, Q_IMAGE_PATH, G_IMAGE_PATH, Q_MAT_DATA, G_MAT_DATA):
    extractor  = pfextractor(MODEL_PATH)
    
    if not os.path.exists(Q_MAT_DATA): 
        os.makedirs(Q_MAT_DATA)
    if not os.path.exists(G_MAT_DATA): 
        os.makedirs(G_MAT_DATA)
        
    # query
    if args.generate_querymat_file:
        q_image_list = os.listdir(Q_IMAGE_PATH)
        q_image_list.sort()
        q_image_num = len(q_image_list)

        Q_IMAGE_MAT_PATH = Q_MAT_DATA+'/'+'query_image_mat.json'
        with open(Q_IMAGE_MAT_PATH, "w") as fq1:
            json.dump(q_image_list, fq1)

        i = 0
        q_features = []
        for q_image_name in q_image_list:
            q_image_path = Q_IMAGE_PATH+'/'+q_image_name
            q_image = cv2.imread(q_image_path)
            q_feature = extractor.extract(q_image)
            q_features.append(q_feature)
            i += 1
            print("query feature: %d/%d, %s is done!" %(i, q_image_num, q_image_name))
        
        Q_FEATURE_MAT_PATH = Q_MAT_DATA+'/'+'query_feature_mat.json'
        with open(Q_FEATURE_MAT_PATH, "w") as fq2:
            json.dump(q_features, fq2)
        
        print("Generating query matrix file completed!")
    
    # gallery
    if args.generate_gallerymat_file:
        singel_G_IMAGE_PATH = os.listdir(G_IMAGE_PATH)
        singel_G_IMAGE_PATH.sort()
        #print(singel_G_IMAGE_PATH)
        for singel_G_IMAGE in singel_G_IMAGE_PATH:
            if not os.path.exists(G_MAT_DATA+'/'+singel_G_IMAGE): 
                os.makedirs(G_MAT_DATA+'/'+singel_G_IMAGE)
            
            g_camera_list = os.listdir(G_IMAGE_PATH+'/'+singel_G_IMAGE)
            g_camera_list.sort()
            print(g_camera_list)
            g_camera_num = len(g_camera_list)
            n = 0
            for g_camera in g_camera_list:
                g_image_list = []
                n += 1

                if not os.path.exists(G_MAT_DATA+'/'+singel_G_IMAGE+'/'+g_camera): 
                    os.makedirs(G_MAT_DATA+'/'+singel_G_IMAGE+'/'+g_camera)

                g_image_list1 = os.listdir(G_IMAGE_PATH+'/'+singel_G_IMAGE+'/'+g_camera)
                
                
                for i in g_image_list1:
                    j = G_IMAGE_PATH+'/'+singel_G_IMAGE+'/'+g_camera+'/'+i
                    im = cv2.imread(j)
                    if im is not None:
                        #print('yes')
                        g_image_list.append(i)
                    else:
                        print('no')

                g_image_num = len(g_image_list)
                
                G_IMAGE_MAT_PATH = G_MAT_DATA+'/'+singel_G_IMAGE+'/'+g_camera+'/'+'gallery_image_mat'+'_'+g_camera+'.json'
                with open(G_IMAGE_MAT_PATH, "w") as fg1:
                    json.dump(g_image_list, fg1)

                j = 0
                g_features = []
                for g_image_name in g_image_list:
                    j += 1
                    g_image_path = G_IMAGE_PATH+'/'+singel_G_IMAGE+'/'+g_camera+'/'+g_image_name
                    g_image = cv2.imread(g_image_path)
                    g_feature = extractor.extract(g_image)
                    g_features.append(g_feature)
                    print("camera: %d/%d, gallery feature: %d/%d" %(n, g_camera_num, j, g_image_num))
            
                G_FEATURE_MAT_PATH = G_MAT_DATA+'/'+singel_G_IMAGE+'/'+g_camera+'/'+'gallery_feature_mat'+'_'+g_camera+'.json'
                with open(G_FEATURE_MAT_PATH, "w") as fg2:
                    json.dump(g_features, fg2)
                
                print("Generating gallery matrix file completed! , camera:%s, %d/%d" %(g_camera, n, g_camera_num))
    
    return 0


def Matching(Q_MAT_DATA, G_MAT_DATA,THRESHOLD):
    single_g_mat_path = os.listdir(G_MAT_DATA)
    single_g_mat_path.sort()
    for single_g_mat in single_g_mat_path:
        g_camera_list = os.listdir(G_MAT_DATA+'/'+single_g_mat)
        g_camera_list.sort()
        q_feature_mat_path = Q_MAT_DATA+'/'+'query_feature_mat.json'
        q_image_mat_path = Q_MAT_DATA+'/'+'query_image_mat.json'
        print("time "+single_g_mat+" : start matching...... ")
        # loda data
        with open(q_feature_mat_path,"r") as fq11:
            q_feature_data = json.load(fq11)
        with open(q_image_mat_path,"r") as fq12:
            q_image_data = json.load(fq12)
        
        for i in range(len(q_image_data)):
            #save_path = "result/%s" %q_image_data[i].split('.')[1]
            
            #if not os.path.exists(save_path): 
                #os.makedirs(save_path)

            q_image   = q_image_data[i]
            q_fea     = q_feature_data[i]
            q_feature = np.array(q_fea)
            q_feature = np.reshape(q_feature, (1, 2048))
            print('ID_'+q_image.split('_')[0]+' matched image :')
            for g_camera in g_camera_list:
                save_path = args.save_path+'/'+'ID_'+q_image.split('_')[0]+'/'+single_g_mat+'_result/'+g_camera
                print(g_camera+"******************************")
                g_feature_mat_path = G_MAT_DATA+'/'+single_g_mat+'/'+g_camera+'/'+'gallery_feature_mat'+'_'+g_camera+'.json'
                g_image_mat_path = G_MAT_DATA+'/'+single_g_mat+'/'+g_camera+'/'+'gallery_image_mat'+'_'+g_camera+'.json'
                
                # loda data
                with open(g_feature_mat_path,"r") as fg11:
                    g_feature_data = json.load(fg11)
                with open(g_image_mat_path,"r") as fg12:
                    g_image_data = json.load(fg12)
                    #print(g_feature_data)
                if g_feature_data != []:
                    distmat = (cdist(q_feature, g_feature_data, metric='cosine'))[0]
                    distmat_np = np.array(distmat)
                    indices = np.argsort(distmat_np)
                    distmat.sort()
                    
                    if len(distmat) >= 8:
                        rank = 8
                    else:
                        rank = len(distmat)
                    for s in range(rank):
                        if distmat[s] < THRESHOLD:
                            temp = indices[s]
                            target_image = g_image_data[temp]
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            #print("matching "+target_image)
                            Savefile(target_image, g_camera, save_path, q_image,single_g_mat)
                else:
                    print('There is no bbox in %s camera.'%g_camera)
                    
            print("%s matching is completed!" %q_image_data[i])
        print("============================================time %s matching is completed!"%single_g_mat)

    return 0


def Savefile(image_name, g_camera, save_path, q_image,single_g_mat):

    src_image_path = G_IMAGE_PATH+'/'+single_g_mat+'/'+g_camera+'/'+image_name
    #dst_image_path = save_path+'/'+'9'+'{:0>3d}'.format(int(q_image.split('_')[0]))+'_'+image_name.split('_')[0]+'_'+ image_name.split('_')[1]+'_'+ str(random.randint(0, 500))
    dst_image_path = save_path+'/'+'9'+'{:0>3d}'.format(int(q_image.split('_')[0]))+'_'+image_name
    shutil.copyfile(src_image_path, dst_image_path)
    
    return 0


class pfextractor():
    def __init__(self, model_path):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU index
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
    
    THRESHOLD = 0.35 # 0.9
    #THRESHOLD = 0.4

    # path
    #Q_IMAGE_PATH = "query/10_00_query"
    #G_IMAGE_PATH = "../gallery_img/10-40"
    #MODEL_PATH   = "model/PED_EXT_008.pt"
   
    #Q_MAT_DATA = "mat/query/10_00_query"
    #G_MAT_DATA = "mat/gallery/10-40_gallery"
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--query_img',type=str,default="query/ID_0-10_query" ,help='query image path')
    parser.add_argument('--gallery_img',default="../gallery_img" ,type=str,help='gallery image path')
    parser.add_argument('--model',type=str,default="model/model.pt" ,help='model path')
    parser.add_argument('--query_mat',type=str,default="mat/query/ID_0-10_query",help='query mat path')
    parser.add_argument('--gallery_mat',type=str,help='gallery mat path')
    parser.add_argument('--save_path',type=str,help='gallery mat path')
    parser.add_argument('--generate_gallerymat_file',action="store_true",help='generate_gallerymat_file function')
    parser.add_argument('--generate_querymat_file',action="store_true",help='generate_querymat_file function')
    parser.add_argument('--matching',action="store_true",help='matching function')
    args         = parser.parse_args()
    
    Q_IMAGE_PATH  = args.query_img
    G_IMAGE_PATH  = args.gallery_img
    MODEL_PATH    = args.model
    
    Q_MAT_DATA    = args.query_mat
    G_MAT_DATA    = args.gallery_mat

    if args.generate_gallerymat_file or args.generate_querymat_file:
        print("start Generatematfile==========================")
        Generatematfile(MODEL_PATH, Q_IMAGE_PATH, G_IMAGE_PATH, Q_MAT_DATA, G_MAT_DATA)
    if args.matching:
        print("start matching==========================")
        Matching(Q_MAT_DATA, G_MAT_DATA,THRESHOLD)
        
