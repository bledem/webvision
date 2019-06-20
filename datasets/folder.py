# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

"""Modified from original pytorch version. """

import torch.utils.data as data
import torch
import torch.nn as nn
import nltk
import natsort

import zipfile
from PIL import Image
import os
import os.path
import numpy as np
import json
import pickle

nltk.download('punkt')

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset_spec(root_dir, dict_):
    """special version."""
    images = []
    root_dir = os.path.abspath(root_dir)
    class_idx = (root_dir.split("/")[-1]) #erased int
    print(class_idx)

    for fname in natsort.natsorted(os.listdir(root_dir)):

        if (is_image_file(fname)):
            #print("secund", fname)
            path = os.path.join(root_dir, fname) #image path
            item = (path, int(dict_[int(class_idx[1:])]))  #pair (img_path, query)
            images.append(item)

    return images


def make_dataset_gen(root_dir):
    """generic version."""
    images = []
    root_dir = os.path.abspath(root_dir)
    num_classes = 0

    for cls_idx in natsort.natsorted(os.listdir(root_dir)): #query
        d = os.path.join(root_dir, cls_idx)
        if not os.path.isdir(d):
            continue

        for fname in natsort.natsorted(os.listdir(d)): #images
            if is_image_file(fname):
                path = os.path.join(root_dir, cls_idx, fname) #image path
                item = (path, int(cls_idx[1:]))  #pair (img_path, query)
                images.append(item)
                if (int(cls_idx[1:])>num_classes):
                    num_classes=int(cls_idx[1:])

        #num_classes += 1

    return images, num_classes




def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class SpecImageFolder(data.Dataset):
    """ A special data loader where the images are arranged in this way: ::

       root/001.jpg
       root/002.jpg
       root/003.jpg
       ...

       Note: root contains the label, see `__init__` for details.

       Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, dict_=None):
        imgs = make_dataset_spec(root, dict_)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in folders of: "
                               + root + "\nSupported image extensions are: "
                               + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.class_idx = int((root.split("/")[-1])[1:]) #int erased
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class GenImageFolder(data.Dataset): #for training
    """A generic data loader where the images are arranged in this way: ::  

        root/0/xxx.png
        root/0/xxy.png
        root/0/xxz.png

        root/1/123.png
        root/1/nsdf3.png
        root/1/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        imgs, num_classes = make_dataset_gen(root)
        print("num_classes", num_classes)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root +
                               "\nSupported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the query class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #print(path,target)    
        return img, target

    def __len__(self):
        return len(self.imgs)




class DriveData(data.Dataset):
    imgs = []
    __ys = []

    def __init__(self, folder_dataset, dataset_type, transform=None, target_transform=None, queries_=[], dict_={}):
        self.transform = transform
        # Open and load text file including the whole training data
        check=0
        out = 0
        #print("len query", range(len(queries_))) #query
        self.queries = queries_ #list of the query names
        #print("val opening",os.path.join(folder_dataset, dataset_type+'_filelist.txt') )
        with open(os.path.join(folder_dataset, dataset_type+'_filelist.txt')) as f:
            for line in f:
                #if(int(line.split()[1]) in range(0,len(queries_))): #if query
                #if key in queries_:
                # Image path
                self.imgs.append(os.path.join(folder_dataset, dataset_type+'_images_resized', line.split()[0]))        
                # Steering wheel label
                self.__ys.append(int(line.split()[1]))
                check=check+1
                #     #print(int(line.split()[1]) )
                # else:
                #out=out+1
                #     #print(int(line.split()[1]) , "not in queries")
        #print( "We keep" , check , " values out of", len(f), "for validation")
        self.root = folder_dataset
        #self.imgs = imgs
        #self.num_classes = num_classes
        self.target_transform = target_transform

        #self.loader = loader

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        img = img.convert('RGB')
        target = int(self.__ys[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(self.__ys[index])
        # Convert image and label to torch tensors
        #img = torch.from_numpy(np.asarray(img))
        #label = torch.from_numpy(np.asarray(target).reshape([1,1]))
        #print(self.imgs[index])
        return img, target

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.imgs)



def make_dataset_gen_sample(root_dir, folder, dict_):
    """generic version."""
    images = []
    root_dir = os.path.abspath(os.path.join(root_dir, folder))

    #print(sorted(os.listdir(root_dir)))
    for cls_idx in natsort.natsorted(os.listdir(root_dir)): #query
        #print("0", cls_idx)

        d = os.path.join(root_dir, cls_idx)
        if not os.path.isdir(d):       
            continue
        #if (int(cls_idx[1:]) in queries ):
            #num_classes += 1
            #print("first", int(cls_idx[1:]))
        for fname in natsort.natsorted(os.listdir(d)): #images
            if (is_image_file(fname)):
                #print("make_dataset_gen_sample", cls_idx[1:])
                path = os.path.join(root_dir, cls_idx, fname) #image path
                item = (path, int(cls_idx[1:]), int(dict_[int(cls_idx[1:])]))  # (img_path, query, classe)
                images.append([path, int(cls_idx[1:]), int(dict_[int(cls_idx[1:])])])
        # if int(cls_idx[1:])>3:
        #     break
                #print("fname", dict_[int(cls_idx[1:])])
                #if (int(dict_[int(cls_idx[1:])])>num_classes):
                    #num_classes=dict_[int(cls_idx[1:])]


    print("dataset is", len(images))
    return images


class SampleData(data.Dataset):


    def __init__(self, root, folder, transform=None, target_transform=None,
                 loader=default_loader, queries_=[], dict_={}):
        self.transform = transform
        self.class_fq = []
        alpha_weight = 0.5
        # Open and load text file including the whole training data
        check=0
        previous=0
        #root_dir = os.path.abspath(root)
        self.queries=queries_
        self.dict=dict_
       
        imgs = make_dataset_gen_sample(root, folder, dict_)

        #Creating the class weights
        #self.class_fq =  [0]*num_classes
        #print("num class", num_classes)
        # for path, target in imgs:
        #     print("target", target)
        #     self.class_fq[target]= self.class_fq[target]+1
        # for elt in self.class_fq:
        #     elt/sum(self.class_fq)
        #     elt=elt  * alpha_weight + (1-alpha_weight)   
        #print("the maximum class for",  folder,"is", num_classes, "with", len(imgs), "imgs")
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root +
                               "\nSupported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
       # self.num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.root = os.path.join(root,folder)

        #self.imgs = imgs
        #self.num_classes = num_classes
        self.target_transform = target_transform

        #self.loader = loader

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the query class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

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

def create_light_dataset(root_img, part):
    dataset=[]
    for k, query_id in enumerate(sorted(os.listdir(root_img))):
        zip_path= os.path.join(root_img, query_id)

        # if (root_img[-1]=="e" and (int(query_id[1:-4])<12590)):
        #     continue
        print("Processing query {} at path {}".format(query_id[:-4], zip_path))
        feature_list=[]#for each query
        filename_list=[]
        if query_id[-3:]== "lst":
            continue
        with zipfile.ZipFile(zip_path, "r", compression=zipfile.ZIP_DEFLATED) as myzip:   #all img of this query 
                    for j in range(0, len(myzip.infolist())):
                        filename_list.append(myzip.infolist()[j].filename)
                        if(filename_list[j][-3:]=="lst"):
                            #print("opening", filename_list[j])
                            with myzip.open(myzip.infolist()[j], "r") as f:
                                image_list = f.readlines()
                            break
                 
                    for idx in range(len(image_list)): #for each img of this query
                        
                        dataset.append( [image_list[idx], idx]) #the path of the image and the position in pickle document
                        #print("create dataset", [image_list[idx], idx])
        if int(query_id[1:-4])>351 and part : break

    return dataset

def extract_query(path): #extract the query name from path
    img_id = os.path.splitext(path)[0].strip('\n') #img id
    query_id = os.path.dirname(img_id) 
    query_id = os.path.splitext(query_id)[0].strip('\n') #query id
    query_id=os.path.basename(query_id)
    #print("query", int(query_id[1:]))
    return int(query_id[1:])


#read until query to determine
def read_light_dataset(root_img, part):
    dataset=[]
    if (root_img[-11:-5] == "google"):
        file_path = os.path.join(root_img, "googl_img_list.lst")
    else:
        file_path = os.path.join(root_img, "flick_img_list.lst")
    with open(file_path, "r") as f:
        line = f.readline()        
        dataset.append([line, f.readline()]) 
        while line :  
            line = f.readline()
            #print("reading", line)
            if (len(line)>2): #valid path 
                query_id = extract_query(line) 
                class_idx = f.readline()   
                if  query_id>part:
                    break
                dataset.append([line, class_idx]) 
                #print("query_id, part", query_id, part)
            
            
            #print("reading dataset", [line, class_idx]) 
    return dataset


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

    with open(os.path.join(save_path, root_img[-11:-5]+ "_img_list.lst"), 'w') as f:
        for line in dataset:
            f.write(line)

    return dataset


def get_query_feat(zip_path, img_name):
    filename_list=[]
    feature_list=[]
    error=False
    with zipfile.ZipFile(zip_path, "r", compression=zipfile.ZIP_DEFLATED) as myzip:   #all img of this query 
        try:
            feature_list.append(torch.from_numpy(pickle.load(myzip.open(img_name+".pkl"))))
            result=torch.cat(feature_list)
        except:
            error=True
            print("in getQueryFeat, error opening", zip_path, img_name+".pkl")
     
    return result


def get_query_feat_temp(path, img_name):
    filename_list=[]
    feature_list=[]
    error=False
    pkl_path = os.path.join(path, img_name+".pkl")
    feature_list.append(torch.from_numpy(pickle.load( open(pkl_path, "rb"))))
    result=torch.cat(feature_list)
 
    return result





def get_caption_feat(root_path, img_name):
    filename_list=[]
    feature=[]    
    # try:
    with open(os.path.join(root_path,img_name+".pkl"), 'rb') as f:
        feature.append(torch(pickle.load(f)))
        feature=torch.cat(feature)
    # except:
    #     print("in getQueryFeat, error opening", os.path.join(root_path,img_name+".pkl"))
        
    return feature

def get_img_feat(zip_path, img_query):
    with zipfile.ZipFile(zip_path, "r", compression=zipfile.ZIP_DEFLATED) as myzip:   #all img of this query 
        with myzip.open(img_query+'.pkl') as f: #open the pickle
            return torch.from_numpy(pickle.load(f))





class EmbeddedFolder(data.Dataset): #for training
    """Load precomputed captions and image features 
    """

    def __init__(self, root_json, root_img, w2v_model, part, transform=None, dict_={}):
        self.vocab=w2v_model.vocab
        
        #self.full_img_dataset=make_dataset_gen_sample(root_img, "",  dict_)

        #self.dataset = create_dataset(root_json, root_img, dict_) #all the json of all query
        
        self.type=root_json[-6:]
        print("type is", self.type)
        self.root_json = root_json
        self.root_img = root_img
        self.dict = dict_
        self.word2idx = dict([(k, v.index) for k, v in w2v_model.vocab.items()])
        self.idx2word = dict([(v, k) for k, v in w2v_model.vocab.items()])
        self.pool =  nn.AdaptiveAvgPool2d((1, 1))
        self.actual_query= ""
        self.actual_folder=""
        self.query_feat = []
        self.transform = transform
        #self.light_dataset = create_light_dataset(root_img, part)
        self.light_dataset = read_light_dataset(root_img, part)
        self.lenght = len(self.light_dataset)

        if len(self.light_dataset) == 0:
            raise(RuntimeError("Extraction failed"))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            
            images features: torch tensor of shape (2048,1,1).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
        """
        #output = self.dataset[index] #img_feat, caption, target, id_

        output = self.light_dataset[index] # img_path, idx in pkl
        #output = self.full_img_dataset[index] #(img_path, query, classe)
        #print("output", output)
#get caption
        img_id= os.path.splitext(os.path.basename(output[0]))[0].strip('\n') #img id
        query_id = os.path.dirname(output[0]) 
        #img_id = os.path.splitext(img_id)[0].decode('utf8').strip('\n')
        query_id = os.path.splitext(query_id)[0].strip('\n') #query id
        query_id=os.path.basename(query_id)
        #print("in getitem", img_id,query_id)

        #if(self.actual_query!=query_id):
        query_feat = get_query_feat(os.path.join(self.root_img,query_id+".zip"), img_id)
        #caption_feat = get_caption_feat(os.path.join(self.root_json,query_id), img_id)

        self.actual_query = query_id
            #self.actual_folder=self.root_img[-6:]
            #print("downloading next query", query_id, "containing", len(self.query_feat), "elt")

        
        sentences = get_caption(self.root_json, query_id,  img_id)
        class_name = int(self.dict[int(query_id[1:])])
        #print("get item", query_id, img_id, class_name)

#get image
        #feat= get_img_feat(os.path.join(self.root_img,query_id+".zip"),img_id )

        #feat= self.query_feat[output[1]]
        #img = Image.open(output[0])
        #img = img.convert('RGB')
        #if self.transform is not None:
        #    img = self.transform(img)

        tokens = nltk.tokenize.word_tokenize(str(sentences).lower())
        not_found=0
        caption=[]
        for word in tokens:
            if len(caption)>256:
                break
            try:
                caption.append( self.word2idx[word] )
            except:
                not_found=not_found+1
       
        if not_found==len(tokens):
            caption.append( self.word2idx["."] )
        #print("get_item img id is", img_id, "and class", class_name, "token size", len(tokens)  )
        caption = torch.LongTensor(caption)
        image = torch.FloatTensor(query_feat)
        image = self.pool(image)
        return image, caption, index, class_name, output[0], query_id, img_id

    def __len__(self):
        return self.lenght




class FusionPrecomp_val(data.Dataset):
    """
    class_limit is the name of the val image until we load the val dataset
    root concat is path to
    """
    def __init__(self, root_concat, class_limit="val262029.pkl", dict_=[]):

            self.avgpool_layer = nn.AdaptiveAvgPool2d((1, 1))
            print("class limit is ", class_limit)
            #list of the pkl concat feat
            self.feat_path_list = create_val_fusion_dataset(root_concat, class_limit) # extract offical image path of the image up to a query
            print("dataset of size", len(self.feat_path_list))
            if len(self.feat_path_list) == 0:
                raise(RuntimeError("Extraction failed"))
            self.root_concat = root_concat
            self.class_limit=class_limit
            self.dict = dict_

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            
            concat resnet and vse feat
        """
        #output = self.dataset[index] #img_feat, caption, target, id_

        feat_path = self.feat_path_list[index] # img_path, idx in pkl
        img_id= os.path.splitext(os.path.basename(feat_path))[0].strip('\n') #img id
        #print("feat path struct in get of val loader", feat_path, img_id)

        for elt in self.dict:
            #print("in val get", elt[0][:-4], img_id)
            if elt[0][:-4]==img_id:
                target=float(elt[1])
                #print(img_id, "target is", target)
                break

        X= pickle.load( open(feat_path, "rb"))
        #img_feat = torch([np.array(elt) for elt in X])
        img_feat_vse= torch.FloatTensor(X[1]).squeeze()
        img_feat_resnet = torch.FloatTensor(X[0]).squeeze()
        img_feat=torch.cat([img_feat_resnet, img_feat_vse])

        #class_name = int(self.dict[int(query_id[1:])])   
        image = torch.FloatTensor(img_feat)
        
        #print("tensor shapes", img_feat.shape, img_feat_vse.shape, img_feat_resnet.shape, target.shape)

        # print("for class {} ".format(class_name))

        return image, target, img_id

    def __len__(self):
        return len(self.feat_path_list)




class FusionPrecomp(data.Dataset):
    def __init__(self, root_resnet, root_vse, class_limit=200, dict_={}):
            self.avgpool_layer = nn.AdaptiveAvgPool2d((1, 1))
            print("class limit is ", class_limit)
            if not class_limit:
                class_limit=13000
            self.type=root_resnet[-11:-5]
            print("type is", self.type)
            #create dataset from resnet feat 
            self.feat_path_list = read_light_dataset(root_resnet, class_limit) # extract offical image path of the images up to a limit query (part arg is the limit)
            print("dataset of size", len(self.feat_path_list))
            if len(self.feat_path_list) == 0:
                raise(RuntimeError("Extraction failed"))
            self.root_resnet = root_resnet
            self.root_vse = root_vse
            self.class_limit=class_limit
            self.dict=dict_

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            
            concat resnet and vse feat
        """
        #output = self.dataset[index] #img_feat, caption, target, id_

        feat_path = self.feat_path_list[index] # img_path, idx in pkl
        img_id= os.path.splitext(os.path.basename(feat_path[0]))[0].strip('\n') #img id
        query_id = os.path.dirname(feat_path[0]) 
        query_id = os.path.splitext(query_id)[0].strip('\n') #query id
        query_id=os.path.basename(query_id)

        #get resnet feat from .pkl inside query zip
        resnet_feat = get_query_feat(os.path.join(self.root_resnet,query_id+".zip"), img_id)
        resnet_feat = torch.squeeze(self.avgpool_layer(resnet_feat))
        #vse_feat = get_query_feat_temp(os.path.join(self.root_vse,query_id+".zip"), img_id)
        # if self.type=="google":
        #     vse_feat = get_query_feat_temp(os.path.join(self.root_vse,query_id), img_id)
        # else:
        vse_feat = get_query_feat(os.path.join(self.root_vse,query_id+".zip"), img_id)
        img_feat= torch.cat([resnet_feat, vse_feat])
        #print("tensor shapes", resnet_feat.shape, vse_feat.shape, img_feat.shape)
        # print("from resnet {} and from vse {}, {} ".format(os.path.join(self.root_resnet,query_id+".zip"), os.path.join(self.root_vse,query_id), img_id))

        self.actual_query = query_id         
        class_name = int(self.dict[int(query_id[1:])])   
        image = torch.FloatTensor(img_feat)
        # print("for class {} ".format(class_name))
        return image, class_name, feat_path[0], query_id, img_id

    def __len__(self):
        return len(self.feat_path_list)




def create_val_fusion_dataset(root_img, class_limit):
    filename_list=[]
    #print(class_limit)
    for elt in natsort.natsorted(os.listdir(root_img)):
        if elt[-3:]=="pkl":
            filename_list.append(os.path.join(root_img, elt))
            #print( elt)
            if elt==class_limit:
                break   

       
    print("val dataset size:", len(filename_list)) 
    return  filename_list


def create_img_dataset( root_img):
    avgpool_layer = nn.AdaptiveAvgPool2d((1, 1))
    imgs=[]
    imgs_path=[]
    filename_list=[]
    len_pkl=[]
    #args path to json query folder and path to feature query zip
    
    zip_path= os.path.join(root_img, "val")
    img_id=0  
    elt=0 
    #with zipfile.ZipFile(zip_path, "r", compression=zipfile.ZIP_DEFLATED) as myzip:   #all img of this query    
    for elt in natsort.natsorted(os.listdir(zip_path)):
        filename_list.append(os.path.join(zip_path, elt))
    filename_list=natsort.natsorted(filename_list)
    #print("filename_list", filename_list)
    for j in range(0, len(filename_list)):  
        
        if(filename_list[j][-8:]=="name.lst"):
            with open(filename_list[j], "r") as f:
                for line in f:
            # Image path
                    #print("line", line[-14:])
                    imgs_path.append(line)
            
                print("imgs_path", len(imgs_path))      
            # Steering wheel label
        elif (filename_list[j][-3:]=="pkl"):
            #print("opening the val feature_set",filename_list[j] )
            with open(filename_list[j], 'rb') as f:
                X = pickle.load(f)
                X =  torch.from_numpy(X)
                #X.cuda()
                #print("Val data X size", len(X), X.shape) #Val data X size 400 torch.Size([400, 2048, 7, 7])
                len_pkl.append(len(X)-1)
                # for i in range(len(X)): #for each img of this query               
                #     x= avgpool_layer(X[i]) 
                #     imgs.append(x) 
                 
    print("Img List size:", len(len_pkl), "total is", len(imgs_path)) 
    return  imgs_path, len_pkl




class PrecompImg(data.Dataset): #for validation
    """Load precomputed captions and image features 
        We load the len of each pkl in order to retrieve the image according to its position
    """


    def __init__(self, root_img):
        self.avgpool_layer = nn.AdaptiveAvgPool2d((1, 1))
        imgs_path, len_pkl = create_img_dataset(root_img)
        sum=0
        for num in len_pkl:
            sum = sum +num
        print("nb of image is", sum, len(imgs_path))
        self.imgs_path=imgs_path
        #self.cls_list=cls_list
        self.len_pkl =len_pkl #list of the size of each pkl
        if len(self.imgs_path) == 0:
            raise(RuntimeError("Extraction failed"))
        self.root_img = root_img
        self.actual_id = 0
        #loading the first pkl
        with open(os.path.join(root_img, "val",'0.pkl'), 'rb') as f:
            X = pickle.load(f)
            X =  torch.from_numpy(X)
            X.cuda()

        self.actual_pkl = X
        self.sub_len=0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            
            images features: torch tensor of shape (2048,1,1).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
        """
        """
        len_pkl is list of the lenght of all pkl from 0.pkl to n.pkl
        actual_id is the actual=id.pkl we are considering now
        """
        if index> self.len_pkl[self.actual_id]+self.sub_len: #we load the next pickle with if the index is inferior
            self.actual_id = self.actual_id+1
            print(index, "LOADING NEW VAL PKL seen from", self.actual_id-1, "to", self.actual_id, " with line from", self.sub_len, "to", self.len_pkl[self.actual_id-1] +self.sub_len ,"included")
            with open(os.path.join(self.root_img , "val",str(self.actual_id)+'.pkl'), 'rb') as f:
                X = pickle.load(f)
                X =  torch.from_numpy(X)
                X.cuda()
            self.actual_pkl=X
            self.sub_len= self.len_pkl[self.actual_id-1] +self.sub_len + 1 #+1 because 
        print("limit for index", len(self.actual_pkl), "index", index, "idx for arrray", index-self.sub_len)
        img=self.actual_pkl[index-self.sub_len]
        x= self.avgpool_layer(img) 

        #print("get", self.imgs_path[index], x.shape )                
        return x, self.imgs_path[index]

    def __len__(self):
        return len(self.imgs_path)


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list (of size batch size) of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, cls_ids, imgs_path, query_id, img_id = zip(*data) #id is item number
    #print("ids", ids, "img id", img_ids)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    #images = torch.FloatTensor(images)
    #len(caption)=batch_size
    #print("caption in collate", len(captions) , len(captions[0]),  len(captions[1]), len(captions[2]),len(captions[10]))
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    #print("max caption lenght of this batch is", max(lengths))
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i] #size of this item
        targets[i, :end] = cap[:end]
        #print( "target", targets[i, :end])
    return images, targets, lengths, ids, cls_ids, imgs_path, query_id, img_id

