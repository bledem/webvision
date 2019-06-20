""" Extract bottleneck features using neural network. """

import os
import math
import sys
import torch
import pickle as cPickle
from os.path import join as pjoin
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from cnn.factory import get_model
from cnn.config import cfg
from datasets.folder import SpecImageFolder, DriveData
from PIL import Image
import numpy as np
from torchsummary import summary
import shutil
import zipfile
import natsort
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] =  "0,3" #we give this number as the device id we use
data_path = "/mnt/data1/webvision2.0"
batch_size = 8 
input_size= 224
arch="resnet50"
gpus ="0,1" 
data_folder ="both" 
save_freq = 500
num_classes= 5607 #number of class we want
print_freq= 10000
load_path ='/home/betty/webvision_train/results/resnet50/5000classes_onemonth/'
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images_so_far = 0
save_path='/mnt/data3/web_feature_training/'
params_file="/mnt/data2/betty/webvision_train/results/resnet50/5000classes_onemonth/model_best.tar"

# queries=[]
# previous=-1
# all=True #to activate if you want all the query put in one class
# if (num_classes is None):
#     all=True
# with open(os.path.join(data_path, 'info', 'queries_synsets_map.txt')) as f:
#         for line in f:
#             # Image path
#             if((line.split()[1] != previous) or all):
#                 queries.append(int(line.split()[0]))
#                 previous=line.split()[1]
#                 #print(line.split()[0], " ", line.split()[1])
#             elif ((len(queries)>=query) and not all):
#                 print("over the class", query, 'queries is', queries)
#                 break


def save_to_pickle(features, save_path, cls_id):

    with open(pjoin(save_path, cls_id + ".pkl"), 'wb') as f:
        cPickle.dump(features, f, protocol=cPickle.HIGHEST_PROTOCOL)

def save_to_txt(img_tuple, save_path, cls_id):
    with open(pjoin(save_path, cls_id, "img_name.lst"), 'w') as f:
        for img_path in img_tuple:
            f.write(img_path.decode('utf8'))


def extract_caption(root_json):
    for j, json_file  in enumerate(sorted(os.listdir(root_json))): #args.data_path  
        print("Processing query {}".format(json_file))
        captions = one_query_json(root_json, json_file) #for one query
        if (len(captions)==0):
            print("Query {} is empty".format(json_file))
            continue
        print("nb of caption for this query is ", len(captions)) 

def extract_img_names(ext_loader, model,
                  save_path):
    save_path = pjoin(save_path, "val")
    model.eval()
    #print(pjoin(save_path, cls_id))
    if not os.path.exists(save_path): #save the resulting feature
        os.mkdir(save_path)

    batch_feat = []
    img_names = ext_loader.dataset.imgs
    # save to text
    with open(pjoin(save_path, "img_name_sub.lst"), 'w') as f:
        for i, img_path in enumerate(img_names):          
            #print(len(cls_id_list),i) 
            f.write(img_path + '\n')
            if (img_path == "/mnt/data1/webvision2.0/val_images_resized/val014228.jpg" ):
                break


def extract_feats_val(ext_loader, model,
                  save_path,
                  save_freq=50):
    # switch to evaluate mode
    save_path = pjoin(save_path, "val")
    model.eval()
    #print(pjoin(save_path, cls_id))
    if not os.path.exists(save_path): #save the resulting feature
        os.mkdir(save_path)

    cls_id_list = []
    batch_feat = []
    img_names = ext_loader.dataset.imgs
    batch_size = ext_loader.batch_size
    num_img = len(img_names)
    init_idx = 0
    pkl_idx = 0
    last_idx = int(math.ceil(num_img / float(batch_size))) - 1

    for idx, data in enumerate(ext_loader, 0):
        
        inputs, cls_id = data

        cls_id_list= cls_id_list+  cls_id.data.tolist()
        print("cls_id", cls_id.data.tolist()[0])

        inputs = Variable(inputs.to(device))
        feats = model(inputs)
        #print("check shape", feats.shape)

        cpu_feat = feats.data.cpu().numpy()
        if len(cpu_feat.shape) == 1:
            cpu_feat = np.reshape(cpu_feat, (1, -1))
        batch_feat.append(cpu_feat)

        if idx % save_freq == (save_freq - 1):
            print("saving pickle", len(batch_feat))
            batch_im_list = img_names[
                init_idx: batch_size * save_freq + init_idx]
            init_idx = batch_size * save_freq + init_idx
            batch_feat = np.concatenate(batch_feat, axis=0)
            # with open(pjoin(save_path, str(pkl_idx) + ".pkl"), 'wb') as f:
            #     cPickle.dump(batch_feat, f, protocol=cPickle.HIGHEST_PROTOCOL)

            batch_feat = []
            pkl_idx += 1
               

        if idx == last_idx:
            batch_im_list = img_names[init_idx:]
            batch_feat = np.concatenate(batch_feat, axis=0)
            print("saving final in ", save_path+str(pkl_idx))
            # with open(pjoin(save_path, str(pkl_idx)+".pkl"), 'wb') as f:
            #     cPickle.dump(batch_feat, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # save to text
    with open(pjoin(save_path, "img_name.lst"), 'w') as f:
        for i, img_path in enumerate(batch_im_list):          
            #print(len(cls_id_list),i) 
            f.write(img_path +" "+ str(cls_id_list[i])+ '\n')
    print("Making archive")
    #shutil.make_archive(pjoin(save_path), 'zip', pjoin(save_path))
    print("Deleting archive")
    #shutil.rmtree(pjoin(save_path))  





def extract_feats_val_solo(ext_loader, model,
                  save_path,
                  save_freq=50):
    # switch to evaluate mode
    save_path = pjoin(save_path, "val")
    model.eval()
    #print(pjoin(save_path, cls_id))
    if not os.path.exists(save_path): #save the resulting feature
        os.mkdir(save_path)

    cls_id_list = []
    batch_feat = []
    img_names = ext_loader.dataset.imgs
    batch_size = ext_loader.batch_size
    num_img = len(img_names)
    init_idx = 0
    pkl_idx = 0
    last_idx = int(math.ceil(num_img / float(batch_size))) - 1

    for idx, data in enumerate(ext_loader, 0):
        
        inputs, cls_id = data

        cls_id_list= cls_id_list+  cls_id.data.tolist()
        print("cls_id", cls_id.data.tolist()[0])

        inputs = Variable(inputs.to(device))
        feats = model(inputs)
        #print("check shape", feats.shape)

        cpu_feat = feats.data.cpu().numpy()
        if len(cpu_feat.shape) == 1:
            cpu_feat = np.reshape(cpu_feat, (1, -1))
        batch_feat.append(cpu_feat)

        if idx % save_freq == (save_freq - 1):
            print("saving pickle", len(batch_feat))
            batch_im_list = img_names[
                init_idx: batch_size * save_freq + init_idx]
            init_idx = batch_size * save_freq + init_idx
            batch_feat = np.concatenate(batch_feat, axis=0)
            # with open(pjoin(save_path, str(pkl_idx) + ".pkl"), 'wb') as f:
            #     cPickle.dump(batch_feat, f, protocol=cPickle.HIGHEST_PROTOCOL)

            batch_feat = []
            pkl_idx += 1
               

        if idx == last_idx:
            batch_im_list = img_names[init_idx:]
            batch_feat = np.concatenate(batch_feat, axis=0)
            print("saving final in ", save_path+str(pkl_idx))
            # with open(pjoin(save_path, str(pkl_idx)+".pkl"), 'wb') as f:
            #     cPickle.dump(batch_feat, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # save to text
    with open(pjoin(save_path, "img_name.lst"), 'w') as f:
        for i, img_path in enumerate(batch_im_list):          
            #print(len(cls_id_list),i) 
            f.write(img_path +" "+ str(cls_id_list[i])+ '\n')
    print("Making archive")
    #shutil.make_archive(pjoin(save_path), 'zip', pjoin(save_path))
    print("Deleting archive")
    #shutil.rmtree(pjoin(save_path))       

def divide_feats(root_path, query_id):
    filename_list=[]
    error=False

    #for one query
    save_path= root_path+"_solo"
    save_query=pjoin(save_path, query_id[:-4])
    feature_nb=0
    image_name=[]
    image_list=[]


    if not os.path.exists(save_query): #save the resulting feature
        os.mkdir(save_query)
    with zipfile.ZipFile(pjoin(root_path, query_id), "r", compression=zipfile.ZIP_DEFLATED) as myzip:   #all img of this query 
        for j in range(0, len(myzip.infolist())):
                        filename_list.append(myzip.infolist()[j].filename)
                        if(filename_list[j][-3:]=="lst"):
                            with myzip.open(myzip.infolist()[j], "r") as f:
                                image_list = f.readlines()
        for img_path in image_list:
            image_name.append( os.path.basename(os.path.splitext(img_path)[0].decode('utf8').strip('\n')))
            #print(image_name)
        filename_list=natsort.natsorted(filename_list)

        for j in range(0, len(filename_list)):   
            if(filename_list[j]=="0.pkl"):
                with myzip.open(filename_list[j]) as f: #open the pickle
                   # try:
                    X=pickle.load(f)
                    print("shape", torch.from_numpy(X).shape)
                    for idx, elt in enumerate(X):
                        save_to_pickle(elt, save_query, image_name[idx])
                        feature_nb=feature_nb+1
                    # except:
                    #     print("error opening", filename_list[j])


        if (len(image_name)!=feature_nb):
            print("error size: ", len(image_name), feature_nb)
        else:
            save_to_txt(image_list, save_path, query_id[:-4])
            shutil.make_archive(pjoin(save_path, query_id[:-4]), 'zip', pjoin(save_path, query_id[:-4]))
            shutil.rmtree(pjoin(save_path, query_id[:-4]))    

def extract_feats(ext_loader, folder, model,
                  save_path, cls_id,
                  save_freq=700):
    # switch to evaluate mode
    save_path = pjoin(save_path, folder)
    model.eval()
    #print(pjoin(save_path, cls_id))
    if not os.path.exists(pjoin(save_path, cls_id)): #save the resulting feature
        os.mkdir(pjoin(save_path, cls_id))

    batch_feat = []
    img_names = ext_loader.dataset.imgs
    batch_size = ext_loader.batch_size
    num_img = len(img_names)
    init_idx = 0
    pkl_idx = 0
    last_idx = int(math.ceil(num_img / float(batch_size))) - 1

    for idx, data in enumerate(ext_loader, 0):
        inputs, _ = data
        inputs = Variable(inputs.to(device))
        feats = model(inputs)
        #print("check shape", feats.shape)

        cpu_feat = feats.data.cpu().numpy()
        if len(cpu_feat.shape) == 1:
            cpu_feat = np.reshape(cpu_feat, (1, -1))
        batch_feat.append(cpu_feat)

        if idx % save_freq == (save_freq - 1):
            batch_im_list = img_names[
                init_idx: batch_size * save_freq + init_idx]
            init_idx = batch_size * save_freq + init_idx
            batch_feat = np.concatenate(batch_feat, axis=0)
            save_to_pickle(batch_feat, save_path, cls_id, str(pkl_idx))

            batch_feat = []
            pkl_idx += 1

        elif idx == last_idx:
            batch_im_list = img_names[init_idx:]
            batch_feat = np.concatenate(batch_feat, axis=0)
            print("saving in ", save_path, cls_id)
            save_to_pickle(batch_feat, save_path, cls_id, str(pkl_idx))

    # save to text
    save_to_txt(img_names, save_path, cls_id)
    shutil.make_archive(pjoin(save_path, cls_id), 'zip', pjoin(save_path, cls_id))
    shutil.rmtree(pjoin(save_path, cls_id))    

def extract_model(data_root, gpus=gpus, batch_size=16,
                  params_file="/mnt/data2/betty/webvision_train/results/resnet50/5000classes_onemonth/model_best.tar",
                  num_workers=4, num_classes=5000,
                  in_size=224, save_path=None, dict_=None):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    ext_transform = transforms.Compose([
        transforms.Resize((in_size, in_size)),
        transforms.ToTensor(),
        normalize])
    #print(params_file)
    assert os.path.isfile(params_file), "{} is not exist.".format(params_file)
    # define the model
    model = get_model(name=arch,
                      num_classes=num_classes, extract_feat=True)
    if len(gpus) > 1:
        prev_gpus = gpus
        gpus = [int(i) for i in gpus.strip('[]').split(',')]
        print("Let's use", gpus, "on", torch.cuda.device_count(), "GPUs!")
        #os.environ["CUDA_VISIBLE_DEVICES"] =  prev_gpus #we give this number as the device id we use
        gpus_idx = range(0,len(gpus))
        model = torch.nn.DataParallel(model, device_ids=gpus_idx)
    elif len(gpus) == 1:
        prev_gpus = gpus
        gpus = [int(i) for i in gpus.strip('[]').split(',')]
        print("Let's use", gpus, "on", torch.cuda.device_count(), "GPUs!")
        #os.environ["CUDA_VISIBLE_DEVICES"] =  prev_gpus #we give this number as the device id we use

    else:
        #os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print("no gpus")

    #model.to(device)
    model.cuda()
    params = torch.load(params_file)

    model.load_state_dict(params['state_dict'])
    #print("feature_map", model)

    modules = list(model.module.children())[:-1]
    #summary(model, input_size=(3, 224, 224))

    #model =  nn.Sequential(*modules)
    
    ##un comment to PRINT the model
#     print("Params to learn:")
#     feature_extract=False
#     if feature_extract:
#         params_to_update = []
#         for name,param in model.named_parameters():
#             if param.requires_grad == True:
#                 params_to_update.append(param)
#                 print("\t",name)
#     else:
    # for name,param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print("\t",name)
    #feature_map = list(model.module.children())

    modules.pop()
    model = nn.Sequential(*modules)
    model.to(device)
    model.cuda()
    print("feature_map2", model)

    # for name,param in model.named_parameters():
    #         if param.requires_grad == True:
    #             print("\t",name)
    #summary(model, input_size=(3, 224, 224))
    
#for every query
    if(data_root[-1]=="d"):
       print("Training data")
       for cls_id in sorted(os.listdir(data_root)): #args.data_path  
           print("Processing class {}".format(cls_id))
           ext_data = SpecImageFolder(root=pjoin(data_root, cls_id),
                                       transform=ext_transform, dict_=dict_)
           ext_loader = DataLoader(dataset=ext_data, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers,
                                    pin_memory=True)
           extract_feats(ext_loader, data_root[24:30], model, save_path,  cls_id) 
    else:

        print("Validation data")
        valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(in_size),
        transforms.ToTensor(),
        normalize])

        valid_data = DriveData(folder_dataset=data_root, dataset_type='val',  #val_images_resized
                                transform=valid_transform, dict_=dict_)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
        print(len(valid_loader)," img in the dataset")
        #extract_img_names(valid_loader, model,save_path)
        extract_feats_val(valid_loader, model, save_path)


query_synset_dic = {}
with open("/mnt/data2/betty/Pictures/webvision/info/queries_synsets_map.txt") as f:
    for line in f:
       (key, val) = line.split()
       query_synset_dic[int(key)] = val
        
# google_path="/mnt/data1/webvision2.0/google_images_resized"
# flickr_path="/mnt/data1/webvision2.0/flickr_images_resized"
val_path="/mnt/data1/webvision2.0"

# save_path_google="/mnt/data3/web_feature_training/google"
# save_path_flickr="/mnt/data3/web_feature_training/flickr"
# print(save_path_flickr)

# for query_zip  in sorted(os.listdir(save_path_google)):
#     if(int(query_zip[1:-4])>=12542) :

#         print("Processing query {}".format(query_zip))
#         divide_feats(save_path_google, query_zip)
#     else:
#         continue

# for query_zip  in sorted(os.listdir(save_path_flickr)):
#     print("Processing query {}".format(query_zip))
#     divide_feats(save_path_flickr, query_zip)

extract_model(data_root=val_path,
                  gpus=gpus,
                  batch_size= batch_size,
                  params_file=params_file,
                  num_workers=num_workers,
                  num_classes=num_classes,
                  in_size= input_size,
                  save_path= save_path,dict_=query_synset_dic)