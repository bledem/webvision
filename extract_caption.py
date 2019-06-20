""" Extract bottleneck features using neural network. """

import os
import math
import sys
import torch
import pickle as cPickle
from os.path import join as pjoin
import torch.nn as nn
import gensim
import nltk

from PIL import Image
import numpy as np

import shutil
import zipfile
import natsort
import pickle
import json

data_path = "/mnt/data1/webvision2.0"
batch_size = 8 
input_size= 224
arch="resnet50"
gpus ="0,1" 
data_folder ="both" 
save_freq = 5000
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



def get_caption(root_json, query_id, img_id):
    full_path = os.path.join(root_json, query_id)
    #print(root_json, query_id, full_path)

    dataset = json.load(open(full_path+".json", 'r'))
    #captions = []
    caption = [obj for obj in dataset if obj['id']==img_id][0]
    if (root_json[-1]=="e"):
        captions = caption['description'] + ' ' +  caption['title']
    else:
        captions =  caption['description'] + ' ' +  caption['title'] + ' ' + caption['tags']
    #print("captions solo", captions)
    #print("extracting json for for", root_json[-6:] )    

    return captions 

def one_query_json(root_json, json_file):
    """ Return a list of pair with title, description """
    full_path = os.path.join(root_json, json_file)
    dataset = json.load(open(full_path, 'r'))
    captions = {}
    for i, d in enumerate(dataset):
        #print (d['id'])
        if (root_json[-1]=="e"):
            captions[d['id']] = d['description'] + ' ' +  d['title']
        else:
            captions[d['id']] = d['description'] + ' ' +  d['title'] + ' ' + d['tags']

        #print("description", captions[d['id']])
    print("extracting json for for", root_json[-6:-1] )    
    return captions





def open_json(root_json, word2idx, save_path):
        for json_file in sorted(os.listdir(root_json)):
            full_path = os.path.join(root_json, json_file)
            query_name = json_file[:-5]
            save_this_query = os.path.join(save_path, query_name)
            print("creating", save_this_query)

            if not os.path.exists(save_this_query): #save the resulting feature
                os.mkdir(save_this_query)

            dataset = json.load(open(full_path, 'r'))
            for i, caption_ in enumerate(dataset): #inside one query
                #print (caption_['id'])
                not_found=0
                img_name=caption_['id']
                if (root_json[-1]=="e"):
                    captions = caption_['description'] + ' ' +  caption_['title']
                else:
                    captions =  caption_['description'] + ' ' +  caption_['title'] + ' ' + caption_['tags']
                

                tokens = nltk.tokenize.word_tokenize(str(captions).lower())
       
                caption=[]
                for word in tokens:
                    if len(caption)>256:
                        #print("discarded", self.root_img,query_id, "nb", index )
                        break
                    try:
                        caption.append(word2idx[word] )
                        #print("idx", self.word2idx[word])
                    except:
                        #print(word, "not found")
                        not_found=not_found+1

            
                if not_found==len(tokens):
                    caption.append( word2idx["."] )
                if i%20==0:
                    print("taking", json_file, "token size", len(tokens)  )
                caption = torch.LongTensor(caption)
                save_to_pickle(caption, save_this_query, img_name)
            print("Making archive", save_this_query)
            # shutil.make_archive(pjoin(save_this_query), 'zip', pjoin(save_path))
            # #print("Deleting archive")
            # shutil.rmtree(pjoin(save_this_query))    



# query_synset_dic = {}
# with open("/mnt/data2/betty/Pictures/webvision/info/queries_synsets_map.txt") as f:
#     for line in f:
#        (key, val) = line.split()
#        query_synset_dic[int(key)] = val
        
google_path="/mnt/data1/webvision2.0/google"
flickr_path="/mnt/data1/webvision2.0/flickr"

save_path_google="/mnt/data4/web_feature_training/google_caption"
save_path_flickr="/mnt/data4/web_feature_training/flickr_caption"
print(save_path_flickr)
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('/home/betty/Downloads/wiki-news-300d-1M-subword.vec', binary=False) #(999994, 300) matrix 
print("model loaded")
word2idx = dict([(k, v.index) for k, v in w2v_model.vocab.items()])

open_json(flickr_path, word2idx, save_path_flickr)