# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import SimpleITK as sitk

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms


from dataset import Dataset
import secmyresnetunetplus #

from metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity
import losses
from utils import str2bool, count_params
import joblib
from hausdorff import hausdorff_distance
import imageio
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os
import cv2
import numpy as np

def vis_feature(x, max_num, out_path):
    for i in range(0, x.shape[1]):
        if i >= max_num:
            break
        feature = x[0, i, :, :].view(x.shape[-2], x.shape[-1])
        feature = feature.cpu().numpy()
        feature = 1.0 / (1 + np.exp(-1 * feature))
        feature = np.round(feature * 255)
#         feature_img = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
        str(i)
        name = ("%s"%i + ".png")
#         name = i + ".png"
        
#         print(name)
        dst_path = os.path.join('picture/%s/' %out_path + name)
#         print(dst_path)
        cv2.imwrite(dst_path, feature)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--mode', default=None,
                        help='GetPicture or Calculate')

    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %val_args.name)
#     with joblib.load('models/%s/args.pkl' %val_args.name) as input_file:
#         try:
#             return pickle.load(input_file)
#         except EOFError:
#             return Non


    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    print("=> creating model %s" %args.arch)
    model = secmyresnetunetplus.__dict__[args.arch](args)########myresnetunetplus
    model = model.cuda()

#     #可视化权重
#     parm={}
#     for name,parameters in model.named_parameters():
# #         print(name,':',parameters.size())
# #         fc0 = parameters['resblock1.senet.fc.2.weight'][:,:]
# #         fc2 = parameters['resblock1.senet.fc.2.weight'][:,:]
#         parm[name]=parameters.detach().cpu().numpy()
#     fc0 = parm['resblock1.senet.fc.2.weight']
#     fc2 = parm['resblock1.senet.fc.2.weight'][:,0]
# #     print("fc0",fc0)
# #     print("fc2",fc2)
    
    # Data loading code
    img_paths = glob(r'../test3/*')
    mask_paths = glob(r'../test/label/*')
    
    #print(mask_paths)

    val_img_paths = img_paths
    val_mask_paths = mask_paths

    #train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    #   train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %args.name))
    model.eval()

    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    if val_args.mode == "GetPicture":

        """
        获取并保存模型生成的标签图
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    input = input.cuda()
                    #target = target.cuda()

                    # compute output
                    if args.deepsupervision:
                        output = model(input)[-1]
#                         print("output",output.shape)
                    else:
                        output = model(input)
#                         print("output",output.shape)
#                     print("img_paths[i]:%s" % img_paths[i])
                    output = torch.sigmoid(output).data.cpu().numpy()
                    img_paths = val_img_paths[args.batch_size*i:args.batch_size*(i+1)]
#                     print("output_shape:%s"%str(output.shape))#（12,4,512,512）
#                     print("target",target.shape)
#                     #可视化：
#                     print("y",y)
#                     vis_feature(x0_0,32,"0")
#                     vis_feature(a,32,"1")
# #                     vis_feature(x2_0,128,"2")
# #                     vis_feature(x3_0,256,"3")
# #                     vis_feature(x4_0,256,"4")
#                     vis_feature(input,5,"input")
    
        
                    for i in range(output.shape[0]):
#                         print("output[0]",output.shape[0])
                        
                        npName = os.path.basename(img_paths[i])
                        overNum = npName.find(".npy")
                        rgbName = npName[0:overNum]
                        rgName = npName[0:overNum]
                        rgbName = rgbName  + ".png"
                        npyName = rgName + ".dcm" 
                        rgbPic = np.zeros([521, 521, 3], dtype=np.uint8)
                        npyPic = np.zeros([512,512],dtype=np.uint8)
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[i,0,idx,idy] > 0.5:#红色
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 0
                                    rgbPic[idx, idy, 2] = 0
                                    npyPic[idx, idy] = 1
                                if output[i,1,idx,idy] > 0.5:#蓝色
                                    rgbPic[idx, idy, 0] = 0
                                    rgbPic[idx, idy, 1] = 0
                                    rgbPic[idx, idy, 2] = 255
                                    npyPic[idx, idy] = 2
                                if output[i,2,idx,idy] > 0.5:#黄色
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 255
                                    rgbPic[idx, idy, 2] = 0
                                    npyPic[idx, idy] = 3
                                if output[i,3,idx,idy] > 0.5:#绿色
                                    rgbPic[idx, idy, 0] = 0
                                    rgbPic[idx, idy, 1] = 255
                                    rgbPic[idx, idy, 2] = 0
                                    npyPic[idx, idy] = 4
                                
                                    
                        imsave('output/%s/'%args.name + rgbName,rgbPic)
                        out = sitk.GetImageFromArray(npyPic)
                        sitk.WriteImage(out,'output/%s/B/'%args.name + npyName)

            torch.cuda.empty_cache()
        """
        将验证集中的GT numpy格式转换成图片格式并保存
        """
        print("Saving GT,numpy to picture")
        val_gt_path = 'output/%s/'%args.name + "G/"
        if not os.path.exists(val_gt_path):
            os.mkdir(val_gt_path)
        for idx in tqdm(range(len(val_mask_paths))):
            mask_path = val_mask_paths[idx]
            name = os.path.basename(mask_path)
            overNum = name.find(".npy")
            name = name[0:overNum]
            rgbName = name + ".png"
            npname = name[0:overNum]
            npName = npname + ".dcm"

            npmask = np.load(mask_path)
            out = sitk.GetImageFromArray(npmask)
            sitk.WriteImage(out,'output/%s/A/'%args.name + npName)

            GtColor = np.zeros([npmask.shape[0],npmask.shape[1],3], dtype=np.uint8)
            for idx in range(npmask.shape[0]):
                for idy in range(npmask.shape[1]):
                    #(标签1) 红色
                    if npmask[idx, idy] == 2:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 0
                        GtColor[idx, idy, 2] = 0
#                     #(标签2) 蓝
                    elif npmask[idx, idy] == 2:
                        GtColor[idx, idy, 0] = 0
                        GtColor[idx, idy, 1] = 0
                        GtColor[idx, idy, 2] = 255
                    elif npmask[idx, idy] == 3:#黄
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 255
                        GtColor[idx, idy, 2] = 0
                    #绿
                    elif npmask[idx, idy] == 4:
                        GtColor[idx, idy, 0] = 0
                        GtColor[idx, idy, 1] = 255
                        GtColor[idx, idy, 2] = 0

#             imsave(val_gt_path + rgbName, GtColor)
            imageio.imwrite(val_gt_path + rgbName, GtColor)
           
        print("Done!")



    if val_args.mode == "Calculate":
        """
        计算各种指标:Dice、Sensitivity、PPV
        """
        wt_dices = []
        tc_dices = []
        et_dices = []
        at_dices = []
        
        wt_sensitivities = []
        tc_sensitivities = []
        et_sensitivities = []
        at_sensitivities = []
        
        wt_ppvs = []
        tc_ppvs = []
        et_ppvs = []
        at_ppvs = []
        
        wt_Hausdorf = []
        tc_Hausdorf = []
        et_Hausdorf = []
        at_Hausdorf = []

        wtMaskList = []
        tcMaskList = []
        etMaskList = []
        atMaskList = []
        
        wtPbList = []
        tcPbList = []
        etPbList = []
        atPbList = []

        maskPath = glob("output/%s/" % args.name + "A/*.dcm")
        pbPath = glob("output/%s/" % args.name + "B/*.dcm")
        if len(maskPath) == 0:
            print("请先生成dicom文件!")
            return

        for myi in tqdm(range(len(maskPath))):
            
            mask_image = sitk.ReadImage(maskPath[myi])
            mask = sitk.GetArrayFromImage(mask_image[:,:,0])
#             print("mask.shape",mask.shape)
#             mask = imread(maskPath[myi])
            pb_image = sitk.ReadImage(pbPath[myi])
            pb = sitk.GetArrayFromImage(pb_image[:,:,0])
#             print("-------------")
#             print("name",myi)
#             print("pb",pb.shape)
#             pb = imread(pbPath[myi])

            wtmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            wtpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            tcmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            tcpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            etmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            etpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            
            atmaskregion = np.zeros([mask.shape[0],mask.shape[1]],dtype=np.float32)
            atpbregion = np.zeros([mask.shape[0], mask.shape[1]],dtype=np.float32)

            for idx in range(mask.shape[0]):
                for idy in range(mask.shape[1]):
#                     print("mask.shape",mask.shape)
                    # label1
                    if mask[idx, idy]==1 :
                        wtmaskregion[idx, idy] = 1
                    if pb[idx, idy]==1:
                        wtpbregion[idx, idy] = 1
                    
#                     if mask[idx, idy,0]==255 and mask[idx, idy, 1]==0 and mask[idx, idy,2] == 0:
#                         wtmaskregion[idx, idy] = 1
#                     if pb[idx, idy,0]==255 and pb[idx, idy, 1]==0 and pb[idx, idy,2] == 0:
#                         wtpbregion[idx, idy] = 1
                    # label2
                    if mask[idx, idy]==2 :
                        tcmaskregion[idx, idy] = 1
                    if pb[idx, idy]==2:
                        tcpbregion[idx, idy] = 1
#                     if mask[idx, idy,0]==0 and mask[idx, idy, 1]==0 and mask[idx, idy,2] == 255 :
#                         tcmaskregion[idx, idy] = 1
#                     if pb[idx, idy,0]==0 and pb[idx, idy, 1]==0 and pb[idx, idy,2] == 255:
#                         tcpbregion[idx, idy] = 1
                    # label3
                    if mask[idx, idy]==3 :
                        etmaskregion[idx, idy] = 1
                    if pb[idx, idy]==3:
                        etpbregion[idx, idy] = 1
    
#                     if mask[idx, idy,0]==255 and mask[idx, idy, 1]==255 and mask[idx, idy,2] == 0:
#                         etmaskregion[idx, idy] = 1
#                     if pb[idx, idy,0]==255 and pb[idx, idy, 1]==255 and pb[idx, idy,2] == 0:
#                         etpbregion[idx, idy] = 1
                    #label4
                    if mask[idx, idy]==4 :
                        atmaskregion[idx, idy] = 1
                    if pb[idx, idy]==4:
                        atpbregion[idx, idy] = 1
#                     if mask[idx, idy,0]==0 and mask[idx, idy, 1]==255 and mask[idx, idy,2] == 0:
#                         atmaskregion[idx, idy] = 1
#                     if pb[idx, idy,0]==0 and pb[idx, idy, 1]==255 and pb[idx, idy,2] == 0:
#                         atpbregion[idx, idy] = 1
            #开始计算label1
            dice = dice_coef(wtpbregion,wtmaskregion)
            
            wt_dices.append(dice)
            ppv_n = ppv(wtpbregion, wtmaskregion)
            wt_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
            wt_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
            wt_sensitivities.append(sensitivity_n)
#             print("labe11: ","dice-",dice," ppv-",ppv_n," Hausdorff-",Hausdorff," sensitivity-",sensitivity_n)
            # 开始计算label2
            dice = dice_coef(tcpbregion, tcmaskregion)
            tc_dices.append(dice)
            ppv_n = ppv(tcpbregion, tcmaskregion)
            tc_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
            tc_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
            tc_sensitivities.append(sensitivity_n)
#             print("labe12: ","dice-",dice," ppv-",ppv_n," Hausdorff-",Hausdorff," sensitivity-",sensitivity_n)
            # 开始计算label3
            dice = dice_coef(etpbregion, etmaskregion)
            et_dices.append(dice)
            ppv_n = ppv(etpbregion, etmaskregion)
            et_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
            et_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(etpbregion, etmaskregion)
            et_sensitivities.append(sensitivity_n)
#             print("labe13: ","dice-",dice," ppv-",ppv_n," Hausdorff-",Hausdorff," sensitivity-",sensitivity_n)
            # 开始计算label4
            dice = dice_coef(atpbregion, atmaskregion)
            at_dices.append(dice)
            ppv_n = ppv(atpbregion, atmaskregion)
            at_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(atmaskregion, atpbregion)
            at_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(atpbregion, atmaskregion)
            at_sensitivities.append(sensitivity_n)
#             print("labe14: ","dice-",dice," ppv-",ppv_n," Hausdorff-",Hausdorff," sensitivity-",sensitivity_n)

        print('Lable1 Dice: %.4f' % np.mean(wt_dices))
        print('Lable2 Dice: %.4f' % np.mean(tc_dices))
        print('Lable3 Dice: %.4f' % np.mean(et_dices))
        print('Lable4 Dice: %.4f' % np.mean(at_dices))
        print("=============")
        print('Lable1 PPV: %.4f' % np.mean(wt_ppvs))
        print('Lable2 PPV: %.4f' % np.mean(tc_ppvs))
        print('Lable3 PPV: %.4f' % np.mean(et_ppvs))
        print('Lable4 PPV: %.4f' % np.mean(at_ppvs))
        print("=============")
        print('Lable1 sensitivity: %.4f' % np.mean(wt_sensitivities))
        print('Lable2 sensitivity: %.4f' % np.mean(tc_sensitivities))
        print('Lable3 sensitivity: %.4f' % np.mean(et_sensitivities))
        print('Lable4 sensitivity: %.4f' % np.mean(at_sensitivities))
        print("=============")
        print('Lable1 Hausdorff: %.4f' % np.mean(wt_Hausdorf))
        print('Lable2 Hausdorff: %.4f' % np.mean(tc_Hausdorf))
        print('Lable3 Hausdorff: %.4f' % np.mean(et_Hausdorf))
        print('Lable4 Hausdorff: %.4f' % np.mean(at_Hausdorf))
        print("=============")
      
   
                    

if __name__ == '__main__':
    main( )
