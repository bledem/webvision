import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0" #we give this number as the device id we use

import time
import shutil
import torch
import gensim
import gensim.downloader as api
from gensim.test.utils import datapath
from os.path import join as pjoin
from tensorboardX import SummaryWriter
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
from tool.embedding_tool import AverageMeter,i2t, t2i, LogCollector, encode_data,save_encoding
import pickle as cPickle
from torchvision import transforms, datasets, models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/mnt/data3/web_feature_training/google',
                         help='path to datasets')
    parser.add_argument('--data_name', default='precomp',
                         help='precomp or full')
    parser.add_argument('--vocab_path', default='/home/betty/Downloads/wiki-news-300d-1M-subword.vec',
                         help='Path to saved vocabulary files.')
    parser.add_argument('--vocab_save', default='/home/betty/save.vec',
                         help='Path to saved vocabulary files.')
    parser.add_argument('--margin', default=0.2, type=float,
                         help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                         help='Number of training epochs.')
    parser.add_argument('--batch_size', default=168, type=int,
                         help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                         help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=256, type=int,
                         help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                         help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                         help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                         help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                         help='Initial learning rate.')
    parser.add_argument('--lr_update', default=60, type=int,
                         help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=4, type=int,
                         help='Number of data loader workers.')
    parser.add_argument('--log_step', default=50, type=int,
                         help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                         help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='/mnt/data4/embedding_training_result',
                         help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', #default='/mnt/data4/embedding_training_result/savepoint.pth.tar', type=str, metavar='PATH',
                         help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                         help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                         help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                         help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='resnet50',
                         help="""The CNN used for image encoder
                         (e.g. vgg19, resnet152)""")
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
    parser.add_argument('--save_fq', help='saving backup fq' , default='700', type=int )

    opt = parser.parse_args()
    print(opt)

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


    #device = torch.device('cuda :0,1,2,3' if torch.cuda.is_available() else 'cpu')
    query_synset_dic={}
    with open("/mnt/data2/betty/Pictures/webvision/info/queries_synsets_map.txt") as f:
        for line in f:
            (key, val) = line.split()
            query_synset_dic[int(key)] = val

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

     # Load Vocabulary Wrapper
    print(opt.vocab_path)
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format('/home/betty/Downloads/wiki-news-300d-1M-subword.vec', binary=False) #(999994, 300) matrix 
    #vocab = pickle.load(open(os.path.join(
    #    opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    #w2v_model = api.load("fasttext-wiki-news-subwords-300") #https://github.com/RaRe-Technologies/gensim-data
    #w2v_model = api.load("glove-wiki-gigaword-50")
    vocab = w2v_model.wv
    #word2idx = dict([(k, v.index) for k, v in w2v_model.vocab.items()])
    #idx2word = dict([(v, k) for k, v in w2v_model.vocab.items()])
    
    opt.vocab_size = len(vocab.vocab)
    print("vocab is", opt.vocab_size)
    model = VSE(opt, w2v_model, gpu_id=0)
    #only img
   

    train_dataset_google = EmbeddedFolder( root_json=pjoin(opt.root_json, "google"), root_img=pjoin(opt.root_img, "google_solo"), w2v_model=w2v_model, transform=train_transform, dict_=query_synset_dic, part=False)
    
    #train_loader_google = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= opt.batch_size, pin_memory=True, collate_fn=collate_fn)

    train_dataset_flickr = EmbeddedFolder(root_json=pjoin(opt.root_json, "flickr"), root_img=pjoin(opt.root_img, "flickr_solo"), w2v_model=w2v_model, transform=train_transform, dict_=query_synset_dic, part=False)
    #train_loader_flickr = torch.utils.data.DataLoader(dataset=train_dataset, batch_size= opt.batch_size, pin_memory=True, collate_fn=collate_fn)
    concat_data = torch.utils.data.ConcatDataset(( train_dataset_flickr, train_dataset_google))
    #print("training from both, on" , len(train_dataset_google)+len(train_dataset_flickr) )
    train_loader = torch.utils.data.DataLoader(dataset= concat_data, batch_size= opt.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    evaluate_loader = torch.utils.data.DataLoader(dataset=train_dataset_google, batch_size= opt.batch_size, pin_memory=True, collate_fn=collate_fn, shuffle=False)
 
   
    best_rsum = 0

     # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            #best_rsum = checkpoint['best_rsum']
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
    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, train_loader)

        rsum=best_rsum

        try:
            rsum = validate(opt, evaluate_loader, model, best_rsum)
        except:
            print("val skipped")
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        #is_best = logger.best_epoch
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters}, is_best, prefix=opt.logger_name + '/')
      



def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    writer = SummaryWriter(comment='webvision_embedding')

    # switch to train mode
    model.img_enc.train()
    model.txt_enc.train()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        if opt.reset_train:
            # Always reset to train mode, this is not the default behavior
            model.img_enc.train()
            model.txt_enc.train()

        # measure data loading time
        data_time.update(time.time() - end)
        nb_img=i*len(train_data)

        # make sure train logger is used
        model.logger = train_logger

        #check GRU training
        print("GRU learning", model.txt_enc.rnn.state_dict()["weight_ih_l0"][0:10])
        data_batch = train_data[4]
        label_batch = train_data[3]
        # Update the model
        #loss_value, img_emb, cap_emb = model.train_emb(*train_data)
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #print("weight", writer, loss_value.data.item(), (epoch*len(train_loader)+i))
        #writer.add_scalar('loss', loss_value.data.item(), (epoch*len(train_loader)+i))
        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Nb_img {nb_img:.3f}\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger), nb_img=nb_img))
       

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        # if model.Eiters % opt.val_step == 0:
        #   validate(opt, val_loader, model)
        #writer.close()
        if model.Eiters % opt.save_fq == 0:
            save_checkpoint({
    'epoch': epoch + 1,
    #'query_id': train_loader.dataset.datasets[0].actual_query,
    #'folder': train_loader.dataset.datasets[0].actual_folder,
    'model': model.state_dict(),
    'opt': opt,
    'Eiters': model.Eiters,}, is_best=False, filename='savepoint.pth.tar', prefix=opt.logger_name +  '/')

def validate(opt, val_loader, model, best_rsum):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(
        model, val_loader, opt.log_step, logging.info)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure=opt.measure)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    #image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, measure=opt.measure)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    
    # if(currscore>best_rsum):
    #     nb=0
    #     print("check sizes", len(img_embs),len(cap_embs) )
    #     while (5000*nb<len(img_embs) and nb*5000<len(cap_embs)):
    #         try:
    #             with open(pjoin(opt.logger_name, "img"+str(nb)+".pkl"), 'wb') as f:
    #                 cPickle.dump(img_embs[nb:nb+5000], f, protocol=cPickle.HIGHEST_PROTOCOL)

    #             with open(pjoin(opt.logger_name, "cap"+str(nb)+".pkl"), 'wb') as f:
    #                 cPickle.dump(cap_embs[nb:nb+5000], f, protocol=cPickle.HIGHEST_PROTOCOL)
    #                 nb=nb+1
    #         except:
    #             break           
    return currscore

def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        print("saving at", prefix + filename)



if __name__ == '__main__':


    main()