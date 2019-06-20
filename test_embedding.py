import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] =  "2" #we give this number as the device id we use

import time
import shutil
import torch
import gensim
import gensim.downloader as api
from gensim.test.utils import datapath
from os.path import join as pjoin
from torch.autograd.variable import Variable
#import data
#from vocab import Vocabulary  # NOQA
#from cnn.encoder import VSE
#from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
import numpy as np
import logging
import tensorboard_logger as tb_logger
from datasets.folder import EmbeddedFolder, collate_fn, PrecompImg 
import argparse
from cnn.encoder import VSE, clip_grad_norm
from tool.embedding_tool import AverageMeter,i2t, t2i, LogCollector, encode_data, save_encoding, save_encoding_train
import pickle as cPickle
from torchvision import transforms, datasets, models
import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/mnt/data3/web_feature_training/google',
                         help='path to datasets')
    parser.add_argument('--data_name', default='precomp',
                         help='precomp or full')
    parser.add_argument('--vocab_path', default='/home/betty/Downloads/wiki-news-300d-1M-subword.vec',
                         help='Path to saved vocabulary files.')
    parser.add_argument('--margin', default=0.2, type=float,
                         help='Rank loss margin.')
    parser.add_argument('--grad_clip', default=2., type=float,
                         help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                         help='Size of an image crop as the CNN input.')
    parser.add_argument('--batch_size', default=256, type=int,
                         help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=4, type=int,
                         help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                         help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                         help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='/mnt/data4/embedding_training_result',
                         help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='/mnt/data4/embedding_training_result/save/savepoint.pth.tar', type=str, metavar='PATH',
                         help='path to latest checkpoint (default: none)')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                         help='Initial learning rate.')
    parser.add_argument('--max_violation', action='store_true',
                         help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                         help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                         help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='resnet50',
                         help="""The CNN used for image encoder
                         (e.g. vgg19, resnet152)""")
    parser.add_argument('--lr_update', default=60, type=int,
                         help='Number of epochs to update the learning rate.')
    parser.add_argument('--use_restval', action='store_true',
                         help='Use the restval data for training on MSCOCO.')
    parser.add_argument('--measure', default='cosine',
                         help='Similarity measure used (cosine|order)')
    parser.add_argument('--use_abs', action='store_true',
                         help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                         help='Do not normalize the image embeddings.')
    parser.add_argument('--reset_train', action='store_true',
                         help='Ensure the training is always done in '
                         'train mode (Not recommended).')
    parser.add_argument('--root_img', default='/mnt/data3/web_feature_training/',
                         help='img or feature root folder')
    parser.add_argument('--root_json', default='/mnt/data1/webvision2.0/',
                         help='json root folder')
    parser.add_argument('--gpus', help='GPU id to use',
                        default="0,1,2,3", type=str)
    parser.add_argument('--save_fq', help='saving backup fq' , default='2000', type=int )
    parser.add_argument('--type', help='saving training or validation data' , default='val', type=str )

    opt = parser.parse_args()
    print(opt)

    query_synset_dic={}
    with open("/mnt/data2/betty/Pictures/webvision/info/queries_synsets_map.txt") as f:
        for line in f:
            (key, val) = line.split()
            query_synset_dic[int(key)] = val

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Image transformer
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(opt.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(opt.crop_size),
            transforms.ToTensor(),
            normalize])

 # Load Vocabulary Wrapper
    print(opt.vocab_path)
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('/home/betty/Downloads/wiki-news-300d-1M-subword.vec', binary=False) #(999994, 300) matrix 

    #w2v_model = api.load("glove-wiki-gigaword-50")
    vocab = w2v_model.wv


    opt.vocab_size = len(vocab.vocab)
    print("vocab is", opt.vocab_size)

    #only img
    if (opt.type == "val"):
        print("considering validation data")
        #val_dataset= PrecompImg(root_img=opt.root_img, pkl_limit=59)
        val_dataset= PrecompImg(root_img=opt.root_img)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size= 1, pin_memory=True, shuffle=False)
        

    else:
            

        train_dataset_google = EmbeddedFolder( root_json=pjoin(opt.root_json, "google"), root_img=pjoin(opt.root_img, "google_solo"), w2v_model=w2v_model, transform=train_transform, dict_=query_synset_dic, part=False)
        #train_dataset_flickr = EmbeddedFolder(root_json=pjoin(opt.root_json, "flickr"), root_img=pjoin(opt.root_img, "flickr_solo"), w2v_model=w2v_model, transform=train_transform, dict_=query_synset_dic, part=False)
        #concat_data = torch.utils.data.ConcatDataset(( train_dataset_flickr, train_dataset_google))
        #print("training from both, on" , len(train_dataset_google)+len(train_dataset_flickr) )
        train_loader_google = torch.utils.data.DataLoader(dataset= train_dataset_google, batch_size= opt.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn)
        #train_loader_flickr = torch.utils.data.DataLoader(dataset= train_dataset_flickr, batch_size= opt.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    model = VSE(opt, w2v_model, gpu_id=0)

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    print("current device", torch.cuda.current_device(),  device)

    #model.cuda() #to force
    best_rsum = 0

     # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            #validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
# Train the Model
  

    # evaluate on validation set
    if (opt.type == "val"):
        save_encoding(model, val_loader, opt, os.path.join(opt.logger_name, "val_feat_concat"), concat=True)
    else:
        #save_encoding_train(model, train_loader_flickr,  os.path.join(opt.logger_name, "train_feat", "flickr"))
        save_encoding_train(model, train_loader_google, os.path.join(opt.logger_name, "train_feat", "google"))

    #rsum = validate(opt, val_loader, model, best_rsum)

  

if __name__ == '__main__':


    main()