# -*- coding: utf-8 -*-
# Author: Xiaoming　Qin

"""Predict probability of the groud-truth class."""

import argparse
import os
import sys
import torch
import zipfile
import numpy as np
import torch.nn as nn
from os.path import join as pjoin
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from cnn.factory import get_model
from cnn.config import cfg
from datasets.folder import SpecImageFolder, DriveData, default_loader
from cnn.utils import adjust_lr, save_ckpt, adjust_lr_manual, plot_confusion_matrix, accuracy, AverageMeter, name_of_class, plot_confusion_matrix_2

from PIL import Image
import torch.nn as nn
import pickle
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():

    parser = argparse.ArgumentParser(description='Test a deep neural network')
    parser.add_argument('--data-path', help='path to data root',
                        default='/data/wv-40/train', type=str)
    parser.add_argument('--gpus', help='GPU id to use',
                        default='0', type=str)
    parser.add_argument('--batch-size', help='mini-batch size',
                        default=16, type=int)
    parser.add_argument('--num-workers', help='number of workers',
                        default=4, type=int)
    parser.add_argument('--params-file', help='model to train on',
                        default='model_best.tar', type=str)
    parser.add_argument('--input-size', help='size of model input',
                        default=224, type=int)
    parser.add_argument('--save-path', help='path to save results',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def save_to_txt(img_paths, probs, labels, save_path):
    with open(pjoin(save_path, "pred_result.lst"), 'w') as f:
        for i in iter(range(len(img_paths))):
            im_name = img_paths[i][0].split('/')[-1]
            prob = probs[i]
            label = labels[i]
            f.write("{0} {1:.5f} {2:d}\n".format(im_name, prob, label))


def check_feature(feat_file, data_root, model,pred_transform):
    #summary(model, input_size=(3, 224, 224))

    feature_map = list(model.module.children())[:-1]
    #print(feature_map)
    feature_map.pop()
    #print("second", feature_map)

    model2 = nn.Sequential(*feature_map)
    model2 = torch.nn.DataParallel(model2, device_ids=[0,1,2,3])
    model2.cuda() 
    model2.eval()
    #print("model normal")
    #summary(model, input_size=(3, 224, 224))
    #print("feature extractor without fc and avg_pool")
    #summary(model2, input_size=(3, 224, 224))

    with zipfile.ZipFile(feat_file, "r", compression=zipfile.ZIP_DEFLATED) as myzip:
        #print("info ",myzip.infolist())
        with myzip.open(myzip.infolist()[0]) as f:
            X = pickle.load(f)
            print(torch.from_numpy(X).float().cuda().shape, "and len is ", len(X))#.view(model.module.fc.in_features,-1).shape)
            print("after avg", model.module.avgpool(torch.from_numpy(X[0])).shape ,
                "and flat", model.module.avgpool(torch.from_numpy(X[0])).view(2048,-1).shape) 
                #give the result after avg torch.Size([2048, 1, 1]) and flat torch.Size([2048, 1])

        with myzip.open(myzip.infolist()[1], "r") as f:
            image_list = f.readlines()

    print("unzip", myzip.infolist()[0].filename, myzip.infolist()[1].filename)
    #print("img list", image_list)
   
    

    image_list = [l.decode('utf8').strip('\n') for l in image_list]
    for idx, img_path in enumerate(image_list):
        print(img_path, idx)
        image = default_loader(img_path)
        if pred_transform is not None:
            image = pred_transform(image)
        inputs = image
        inputs = Variable(inputs).cuda()
        inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2)) # add batch dim in the front
        model.eval()
        #print("image size", inputs.shape)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        outputs_to_check = model.module.fc(model.module.avgpool(torch.from_numpy(X[idx])).float().cuda().view(-1, model.module.fc.in_features))
        _, preds_ft = torch.max(outputs_to_check, 1)


        output2 = model2(inputs)
        print("result", preds, preds_ft )
        #print("ft1", torch.from_numpy(X[idx]).float().cuda(), "ft2", output2) #prove that the images are correct
        #print("ft1", outputs_to_check, "ft2"  , outputs)

#    check_feature( feature_file, img_name_file, model,pred_transform)

def create_confusion_mat(model, save_path, query_synset_dic , data_root="/mnt/data2/betty/webvision_train/results/resnet50/training_features/google",  ):
    print("Creating confusion matrix..")
    classes_names, synset_name = name_of_class()

    # switch to evaluate mode
    model.eval()

#for the confusion matrix
    cm_label=np.array([]);
    cm_pred=np.array([]);
    label=np.array([0])
    for cls_id in sorted(os.listdir(data_root)): #args.data_path  
            print("Processing class {}".format(int(cls_id[1:])), " ie class:", query_synset_dic[int(cls_id[1:])])
            label[0] = int(query_synset_dic[int(cls_id[1:])])

            with open(pjoin(data_root, cls_id, "0.pkl"), 'rb') as f:
                X = pickle.load(f)
                print(torch.from_numpy(X[0]).float().cuda().shape)#.view(model.module.fc.in_features,-1).shape)
            with open(pjoin(data_root, cls_id, "img_name.lst"), 'r') as f:
                image_list = f.readlines()

            image_list = [l.strip('\n') for l in image_list]
            for idx, img_path in enumerate(image_list):
                print(img_path, idx)
                outputs_to_check = model.module.fc(torch.from_numpy(X[idx]).float().cuda().view(-1, model.module.fc.in_features))
                _, preds_ft = torch.max(outputs_to_check, 1)

                print("debug ", cm_label, label ,cm_label.shape, label.shape, preds_ft.shape)
                cm_label = np.concatenate([cm_label, label])
                cm_pred = np.concatenate([cm_pred, preds_ft.data.cpu().numpy().astype(int)])
                #np.set_printoptions(suppress=True)
                print('cm_label', cm_label, 'cm_pred', cm_pred )
                cnf_matrix = confusion_matrix(cm_label, cm_pred)

            
                #cnf_matrix = confusion_matrix(cm_label, cm_pred)
                #np.set_printoptions(precision=2)
               #ax = plot_confusion_matrix(cm_label, cm_pred, classes=classes_names,
                          #title='Confusion matrix, without normalization')
        #plt.savefig('/home/betty/webvision_train/results/resnet50/confusion_matrix_cls_names.png')

   
    #Confusion matrix plotting
    cnf_matrix = confusion_matrix(cm_label, cm_pred)
    np.set_printoptions(precision=2)
    ax = plot_confusion_matrix(cm_label, cm_pred, classes=np.arange(5000),
                      title='Confusion matrix, without normalization')
    plt.savefig('/mnt/data2/webvision_train/results/resnet50/conf_mat_training/confusion_matrix.png')
    ax = plot_confusion_matrix(cm_label, cm_pred, classes=classes_names,
                      title='Confusion matrix, without normalization')
    plt.savefig('/mnt/data2/betty/webvision_train/results/resnet50/conf_mat_training/confusion_matrix_cls_names.png')

    #plt.show()

def pred_gt_probs(pred_loader, model, rescale, save_path):
    # switch to evaluate mode
    model.eval()

    p_arr = []
    l_arr = []
    num_samples = 0.
    num_correct = 0.
    img_paths = pred_loader.dataset.imgs
    # ground truth label
    #gt_label = int(cls_id)
    for idx, data in enumerate(pred_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        output = model(inputs)
        probs = rescale(output)
        #gt_prob = probs[:, gt_label].cpu().data.numpy()
        #p_arr.append(gt_prob)

        _, preds = torch.max(output, 1)
        pd_lbl = preds == labels
        pd_lbl = pd_lbl.cpu().data.numpy()
        l_arr.append(pd_lbl)

        # For debug
        correct = (preds == labels).sum()
        num_samples += labels.size(0)
        num_correct += correct.item()

    # print("gt label: {}".format(gt_label))
    print("num correct: {}".format(num_correct))
    print("num samples: {}".format(num_samples))
    print("acc: {0:.4f}".format(num_correct / float(num_samples)))

    #p_arr = np.concatenate(p_arr, axis=0)
    l_arr = np.concatenate(l_arr, axis=0)
    save_to_txt(img_paths, p_arr, l_arr, save_path)

    return num_correct, num_samples


def predict_model(data_root, gpus, batch_size=16,
                  params_file='/mnt/data2/betty/webvision_train/results/resnet50/5000classes_onemonth/model_best.tar',
                  num_workers=4, num_classes=5607,
                  in_size=224, save_path=None):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    query_synset_dic={}
    with open("/mnt/data2//betty/Pictures/webvision/info/queries_synsets_map.txt") as f:
            for line in f:
               (key, val) = line.split()
               query_synset_dic[int(key)] = val

    pred_transform = transforms.Compose([
        transforms.Resize((in_size, in_size)),
        transforms.ToTensor(),
        normalize])
    print(params_file)
    assert os.path.isfile(params_file), "{} is not exist.".format(params_file)
    params = torch.load(params_file)

    # define the model
    model = get_model(name=params['arch'], num_classes=num_classes)
    if len(gpus) > 1:
        prev_gpus = gpus
        gpus = [int(i) for i in gpus.strip('[]').split(',')]
        print("Let's use", len(gpus), "on", torch.cuda.device_count(), "GPUs!")
        os.environ["CUDA_VISIBLE_DEVICES"] =  prev_gpus #we give this number as the device id we use
        gpus_idx = range(0,len(gpus))
        model = torch.nn.DataParallel(model, device_ids=gpus_idx)
        model.to(device)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print("no gpus")
    #model.cuda()
    model.load_state_dict(params['state_dict'])

    #create_confusion_mat(model, "/mnt/data2/betty/webvision_train/results/resnet50/conf_mat_training", query_synset_dic=query_synset_dic,
    # data_root="/mnt/data2/betty/webvision_train/results/resnet50/training_features/google")

    # Operation to get probability
    rescale = nn.Softmax(dim=1)

    num_correct = 0.
    num_samples = 0.

    img_name_file = "/home/betty/webvision_train/results/resnet50/training_features/google/q00001/img_name.lst"
    feature_file = "/mnt/data3/web_feature_training/google/q00001.zip"
    
    check_feature( feature_file, data_root, model,pred_transform)
    check_feat_dataloader (ext_loader,feature_file, img_name_file, model,pred_transform )

    #for cls_id in sorted(os.listdir(data_root)):
    #print("Processing class {}".format(cls_id))
    pred_data = DriveData(folder_dataset=data_root, dataset_type='val',
                                transform=pred_transform)
    pred_loader = DataLoader(dataset=pred_data, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    
    nc, ns = pred_gt_probs(pred_loader, model, rescale, save_path)
    num_correct += nc
    num_samples += ns

    print("ave acc: {0:.5f}".format(num_correct / float(num_samples)))


def main():
    args = parse_args()
    predict_model(data_root=args.data_path,
                  gpus=args.gpus,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  in_size=args.input_size,
                  params_file=args.params_file,
                  save_path=args.save_path)

if __name__ == "__main__":
    main()
