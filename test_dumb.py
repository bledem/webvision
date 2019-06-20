import torch.utils.data as data
import torch
import torch.nn as nn
import natsort
from datasets.folder import EmbeddedFolder, collate_fn, PrecompImg 

import zipfile
from PIL import Image
import os
import os.path
import numpy as np
import json
import pickle
import gensim
import gensim.downloader as api

def write_light_dataset(root_img, save_path):
    dataset=[]
    for k, query_id in enumerate(sorted(os.listdir(root_img))):
        zip_path= os.path.join(root_img, query_id)

        # if (root_img[-1]=="e" and (int(query_id[1:-4])<12590)):
        #     continue
        print("Processing query {} at path {}".format(query_id[:-4], zip_path))
        feature_list=[]#for each query
        filename_list=[]

        with zipfile.ZipFile(zip_path, "r", compression=zipfile.ZIP_DEFLATED) as myzip:   #all img of this query 
                    for j in range(0, len(myzip.infolist())):
                        filename_list.append(myzip.infolist()[j].filename)
                        if(filename_list[j][-3:]=="lst"):
                            #print("opening", filename_list[j])
                            with myzip.open(myzip.infolist()[j], "r") as f:
                                image_list = f.readlines()
                            break
                 
                    for idx in range(len(image_list)): #for each img of this query
                        # img_id= os.path.basename(image_list[idx])
                        # img_id = os.path.splitext(img_id)[0].decode('utf8').strip('\n')
                        # print("id", idx, img_id, image_list[idx])
                        #feat= avgpool_layer(feature_list[i]) 
                        dataset.append( [image_list[idx], idx]) #the path of the image and the position in pickle document

    with open(os.path.join(save_path, root_img[-11:-6]+ "_img_list.lst"), 'w') as f:
        for path, idx in dataset:
            f.write(path.decode('utf8')+" ")
            f.write(str(idx)+ "\n")


    return dataset


def check_data_generation(original_folder, generated_folder):
    print("checkin data")
    original=[]
    generated=[]
    to_del=[]
    list_folder = natsort.natsorted(os.listdir(original_folder))
    for folder in list_folder:
        names = [os.path.splitext(os.path.basename(elt))[0].strip('\n') for elt in natsort.natsorted(os.listdir(os.path.join(original_folder, folder)))]
        original.append(names)
        temp = [(os.path.basename(elt)) for elt in natsort.natsorted(os.listdir(os.path.join(original_folder, folder)))]
        to_del.append(temp)
    # for i, elt in enumerate(to_del):
    #     count=0
    #     for file in elt:
    #         if file[-3:]=="pkl":
    #             count+=1
    #             print("pkl", os.path.join(original_folder, list_folder[i], file), " found", len(file), count)
    #             os.remove(os.path.join(original_folder, list_folder[i], file))
    list_folder = natsort.natsorted(os.listdir(generated_folder))
    for folder in list_folder:
        names= [os.path.splitext(os.path.basename(elt))[0] for elt in natsort.natsorted(os.listdir(os.path.join(generated_folder, folder)))]
        generated.append(names)
    if (len(original) != len(generated)):
        print("error in lenght", len(original), len(generated))

    for idx in range(len(generated)):
        if(original[idx] != generated[idx]):
            print("query from", idx, list_folder[idx], len(original[idx]), len(generated[idx]), "has pb")


check_data_generation( "/mnt/data1/webvision2.0/flickr_images_resized", "/mnt/data4/embedding_training_result/train_feat/flickr/img_feat")
print("finished")
# write_light_dataset("/mnt/data3/web_feature_training/google_solo","/mnt/data3/web_feature_training")
# write_light_dataset("/mnt/data3/web_feature_training/flickr_solo","/mnt/data3/web_feature_training")
# query_synset_dic={}
# with open("/mnt/data2/betty/Pictures/webvision/info/queries_synsets_map.txt") as f:
#     for line in f:
#         (key, val) = line.split()
#         query_synset_dic[int(key)] = val

# w2v_model = api.load("glove-wiki-gigaword-50")
# root_json="/mnt/data1/webvision2.0/"
# root_img="/mnt/data3/web_feature_training/"
# train_dataset_flickr = EmbeddedFolder(root_json=os.path.join(root_json, "flickr"), root_img=os.path.join(root_img, "flickr_solo"), w2v_model=w2v_model, dict_=query_synset_dic, part=False)

# train_dataset_google = EmbeddedFolder( root_json=os.path.join(root_json, "google"), root_img=os.path.join(root_img, "google_solo"), w2v_model=w2v_model, dict_=query_synset_dic, part=False)
    
# #train_loader_flickr = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= opt.batch_size, pin_memory=True, collate_fn=collate_fn)
# concat_data = torch.utils.data.ConcatDataset(( train_dataset_flickr, train_dataset_google))
# #print("training from both, on" , len(train_dataset_google)+len(train_dataset_flickr) )
# train_loader = torch.utils.data.DataLoader(dataset= concat_data, batch_size= 256, pin_memory=True, shuffle=True, collate_fn=collate_fn)

# with zipfile.ZipFile("/mnt/data3/web_feature_training/google/q00116.zip", "r", compression=zipfile.ZIP_DEFLATED) as myzip:   #all img of this query 
#             # with myzip.open(myzip.infolist()[len(myzip.infolist())-1], "r") as f:
#             #             image_list = f.readlines()
#             #             #print("image_list", image_list)
#             # img_id = os.path.splitext(img_id)[0].decode('utf8').strip('\n')
#             print(myzip.namelist())
#             myzip.open('0.pkl', )
#             filename_list=[]
#             # print("img_id", img_id)
#             for j in range(0, len(myzip.infolist())-1):   
#                 #print(myzip.infolist()[j].filename[-3:])
#                 filename_list.append(myzip.infolist()[j].filename)
            #test=['img_2.lst', '1.pkl', '0.pkl', '2.pkl', 'img_name.lst']
            #print("before", myzip.infolist(), filename_list)
            #print("after", natsort.natsorted(filename_list))
            #break
            # with myzip.open(myzip.infolist()[j]) as f:
            #     X = pickle.load(f)
            #     X =  torch.from_numpy(X)
            # with myzip.open(myzip.infolist()[j]) as f:
            #     try:
            #         X = pickle.load(f)
            #         X =  torch.from_numpy(X)
            #         X.cuda()
            #     except:
            #         continue
            #     print("X size", len(X), X.shape)
            #     for i in range(len(X)): #for each img of this query
            #         img_id= os.path.basename(image_list[i])
            #         img_id = os.path.splitext(img_id)[0].decode('utf8').strip('\n')
            #         #print("id", img_id)
            #         x= avgpool_layer(X[i]) 
            #         dataset.append( [ x,captions[img_id], int(dict_[int(json_file[1:-5])]), img_id]) 
            
            #     print("Img List size:", len(image_list), " and captions size: ", len(captions)) 
            #     print("check captions", img_id, " ", captions[img_id])