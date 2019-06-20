import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "0,1,2, 3" #we give this number as the device id we use

import time
import shutil
import torch
import gensim
import gensim.downloader as api
from gensim.test.utils import datapath
from os.path import join as pjoin
from tensorboardX import SummaryWriter
from torch.autograd.variable import Variable
import torch.nn as nn
import torch.optim as optim
#import data
#from vocab import Vocabulary  # NOQA
#from cnn.encoder import VSE
#from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
import numpy as np
import logging
import tensorboard_logger as tb_logger
from datasets.folder import   PrecompImg, FusionPrecomp, FusionPrecomp_val
import argparse
from cnn.encoder import VSE, clip_grad_norm, FusionNet_three
from cnn.utils import accuracy
from tool.logger import Logger
from tool.embedding_tool import AverageMeter,i2t, t2i, LogCollector, encode_data,save_encoding
import pickle as cPickle
from torchvision import transforms, datasets, models
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', default=30, type=int,
                         help='Number of training epochs.')
    parser.add_argument('--batch_size', default=256, type=int, #512
                         help='Size of a training mini-batch.')
    parser.add_argument('--original_root', default="/mnt/data1/webvision2.0", type=str, help="orininal location of webvision2.0 dataset")
    parser.add_argument('--embed_size', default=256, type=int,
                         help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                         help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=0.05, type=float,
                         help='Initial learning rate.')
    parser.add_argument('--lr_update', default=1, type=int,
                         help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=4, type=int,
                         help='Number of data loader workers.')
    parser.add_argument('--log_step', default=50, type=int,
                         help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=50, type=int,
                         help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='/mnt/data4/fusion_training_result/log',
                         help='Path to save  Tensorboard log.')
    parser.add_argument('--save_path', default='/mnt/data4/fusion_training_result/',
                         help='Path to save the model ')
    parser.add_argument('--resume', #default='/mnt/data4/fusion_training_result/save/savepoint_2864_on_31294.pth.tar', type=str, metavar='PATH',
                         help='path to latest checkpoint (default: none)')
    parser.add_argument('--img_dim', default=2048, type=int,
                         help='Dimensionality of the image embedding.')
    parser.add_argument('--use_abs', action='store_true',
                         help='Take the absolute value of embedding vectors.')
    parser.add_argument('--root_resnet_feat', default='/mnt/data3/web_feature_training/',
                         help='img or feature root folder')
    parser.add_argument('--root_vse_feat', default='/mnt/data4/embedding_training_result/',
                         help='vse root folder')
    parser.add_argument('--save_fq', help='saving backup fq' , default='500', type=int ) #500

    opt = parser.parse_args()
    print(opt)



    query_synset_dic={}
    with open("/mnt/data2/betty/Pictures/webvision/info/queries_synsets_map.txt") as f:
        for line in f:
            (key, val) = line.split()
            query_synset_dic[int(key)] = val
    val_dic=[]
    with open(os.path.join( opt.original_root, "val_filelist.txt")) as f:
        for line in f:
            (val, key) = line.split()
            val_dic.append((val, key))
    #print("val", val_dic)


    tb_logger.configure(opt.logger_name, flush_secs=5)
    logger= Logger(opt.logger_name)
   
    model = FusionNet_three()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    #only img
   
    val_dataset= FusionPrecomp_val(root_concat=pjoin(opt.root_vse_feat, "val_feat_concat"), dict_=val_dic, class_limit=None)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size= opt.batch_size, pin_memory=True, shuffle=True)

    train_dataset_google = FusionPrecomp( root_vse=pjoin(opt.root_vse_feat, "train_feat", "google", "img_feat"), root_resnet=pjoin(opt.root_resnet_feat, "google_solo"),dict_= query_synset_dic, class_limit=None)
    train_dataset_flickr = FusionPrecomp(root_vse=pjoin(opt.root_vse_feat, "train_feat", "flickr", "img_feat"), root_resnet=pjoin(opt.root_resnet_feat, "flickr_solo"), dict_= query_synset_dic, class_limit=None)
    concat_data = torch.utils.data.ConcatDataset(( train_dataset_flickr, train_dataset_google))
    train_loader = torch.utils.data.DataLoader(dataset= concat_data, batch_size= opt.batch_size, pin_memory=True, shuffle=True)
 
   
    best_rsum = 0

     # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            model.module.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))



    criterion = nn.CrossEntropyLoss()
    optimizer =  optim.SGD(model.module.parameters(), lr=opt.learning_rate, momentum=0.9)
# Train the Model
    for epoch in range(opt.num_epochs):
       
        
        # train for one epoch
        adjust_learning_rate(opt, optimizer, epoch)
        train(opt, train_loader, model, epoch, optimizer, criterion, logger)
      
        ave_loss, ave_acc, ave_top5 = validate(val_loader, model, criterion, epoch, logger, opt.val_step, opt.val_step) 
        if ave_top5>best_rsum:
            best_rsum=ave_top5
            is_best= True
        else:
            is_best= False
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'best_rsum': best_rsum,
                'opt': opt,
                'Eiters': model.module.Eiters}, is_best, prefix=opt.save_path + '/')
      



def train(opt, train_loader, model, epoch,  optimizer, criterion, logger):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    writer = SummaryWriter(comment='webvision_embedding')
    

    # switch to train mode
    print("Starting training..")
    model.train()
    
    running_loss, running_acc = 0., 0.
    num_samples, num_correct = 0., 0.
    for i, train_data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        nb_img=i*opt.batch_size
        optimizer.zero_grad()
        inputs, labels = train_data[0].to(device), train_data[1].to(device).long()
        #print("train data", train_data[0].shape, train_data[1].shape )
        model.module.logger = train_logger

        output = model.forward(inputs)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        # if (acc1[0]>1 or acc5[0]>1):
        #     print("acc pb with", acc1[0], acc5[0])
        loss = criterion(output, labels )
        loss.backward()

        value, preds = torch.max(output, 1) #max value among all pred by cnn for each batch img, indice in the row of this max 
        #print("Output max proba, class idx of the highest proba: ", value, preds.data.cpu().numpy())
        top_5_ps, top_5_classes = output.topk(5, dim=1) #top_5_ps.shape is torch.Size([4, 5]). top_5_classes first cl is the top1 pred and other column are top5

        #for kato-san images-all_pred inference
        #for i in range(ps.shape[0]):
        #all_pred_m= np.vstack((all_pred_m, output_soft.data.cpu().numpy()[:,:5000]))
        correct = (preds == labels).sum()

        num_samples += labels.size(0)
        num_correct += correct.item()
        running_loss += loss.item()        
        top_5_classes = top_5_classes.t()
        correct5 =  top_5_classes.eq(labels.view(1, -1).expand_as(top_5_classes))
        correct_5 = correct5[:5].view(-1).float().sum(0, keepdim=True)
        #print("acc1", acc1, "acc5", acc5)
        #print('details', top_5_classes , 'gtruth', labels )
        #print('correct whole', preds, correct)
        #print('correct5 %', correct_5.data, 'and correct top1', correct.data)

        optimizer.step()
        running_loss += loss.item()
        # make sure train logger is used

        batch_time.update(time.time() - end)
        model.module.Eiters +=1
        #print("weight", writer, loss_value.data.item(), (epoch*len(train_loader)+i))
        #writer.add_scalar('loss', loss_value.data.item(), (epoch*len(train_loader)+i))
        # Print log info
        #print("iteration", model.Eiters)
        if model.module.Eiters % opt.log_step == 0:
            running_acc = num_correct / float(num_samples)

            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                '{date} \t'
                'Loss {loss:.3f} \t'
                'Nb_img {nb_img}\t'
                .format(epoch, i, len(train_loader), date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     e_log=str(model.module.logger), nb_img=nb_img, loss= loss.item()))
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                '{date} \t'
                'Loss {loss:.3f} \t'
                'Nb_img {nb_img}\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'
                .format(epoch, i, len(train_loader), date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     e_log=str(model.module.logger), nb_img=nb_img, loss= loss.item(),  top1=top1, top5=top5))

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            # 1. Log scalar values (scalar summary)
            info = { 'loss': loss.item(), 'accuracy1': acc1[0] , 'accuracy5' : acc5[0]}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, model.module.Eiters)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in model.module.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), model.module.Eiters)
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), model.module.Eiters)

            # 3. Log training images (image summary)
            # info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

            # for tag, images in info.items():
            #     logger.image_summary(tag, images, step+1)

            num_samples, num_correct = 0., 0.
            running_loss = 0.

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.module.Eiters)
        tb_logger.log_value('step', i, step=model.module.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.module.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.module.Eiters)
        model.module.logger.tb_log(tb_logger, step=model.module.Eiters)

        if model.module.Eiters % opt.save_fq == 0:
            save_checkpoint({
    'epoch': epoch + 1,
    #'query_id': train_loader.dataset.datasets[0].actual_query,
    #'folder': train_loader.dataset.datasets[0].actual_folder,
    'model': model.state_dict(),
    'opt': opt,
    'optimizer_state_dict': optimizer.state_dict(),
    'Eiters': model.module.Eiters,}, is_best=False, filename='savepoint.pth.tar', prefix=opt.save_path +  '/')
        



def validate(valid_loader, model, criterion, epoch, logger, print_freq=1, save_freq=10000):
    print("Start validation..")
    # switch to evaluate mode
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    num_correct = 0.
    num_samples = 0.
    num_batch = 0
    v_value_loss = 0.
    num_correct5 = 0.


    for idx, data in enumerate(valid_loader, 0):
        num_batch = idx + 1
        inputs, labels, img_id = data 
        #print("labels", labels)      
        cm_label= labels.cpu().numpy()

        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device)) #  labels = Variable(labels.to(device)).cuda() to force on device 3
        labels = labels.long()
        output = model(inputs)
        loss = criterion(output, labels)

        value, preds = torch.max(output, 1)
        value_soft, preds_soft = torch.max(torch.nn.functional.softmax(output, dim=1).data, 1)
        output_soft = torch.nn.functional.softmax(output, dim=1).data
        #ps = torch.exp(output)
        #value, ind = torch.max(ps, 1)
        #print('shape output', ps.shape, 'and preds', preds.shape) #shape output torch.Size([4, 5607]) and preds torch.Size([4]) for batch_size 1 on 4GPU
        top_5_ps, top_5_classes = output.topk(5, dim=1) #top_5_ps.shape is torch.Size([4, 5]). top_5_classes first cl is the top1 pred and other column are top5
        #print('preds debug', preds.data.cpu().numpy())
        # cm_label = np.concatenate([cm_label,labels.data.cpu().numpy()])
        # cm_pred = np.concatenate([cm_pred, preds.data.cpu().numpy().astype(int)])
        #np.set_printoptions(suppress=True)
        #for top5 acc
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        #for kato-san images-all_pred inference
        #for i in range(ps.shape[0]):
        #all_pred_m= np.vstack((all_pred_m, output_soft.data.cpu().numpy()[:,:5000]))

        correct = (preds == labels).sum()
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        top_5_classes = top_5_classes.t()
        correct5 =  top_5_classes.eq(labels.view(1, -1).expand_as(top_5_classes))
        #print('details', top_5_classes , 'gtruth', labels )
        #print('correct whole', preds, correct)

        #print('correct5 whole', correct5)

        correct_5 = correct5[:5].view(-1).float().sum(0, keepdim=True)

        #print('correct5 %', correct_5.data, 'and correct top1', correct.data)

        #cm_pred+= preds.cpu().numpy()

        num_samples += labels.size(0)
        num_correct += correct.item()
        num_correct5 += correct_5.item()
        v_value_loss += loss.item()

        
        if idx % print_freq == (print_freq - 1):
            print( 'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t' 
                 'Acc@5 {top5_val:.3f} ({top5.avg:.3f})\t' .format(top1_val=top1.val/float(labels.size(0)),
                  top5_val=correct_5.item()/float(labels.size(0)),
                  top1=top1, top5=top5))
            #print('soft shape:', output_soft.shape, " ,top value of soft:", value_soft, " ,top index:", preds_soft ," ,gt:", labels)

            #cnf_matrix = confusion_matrix(cm_label, cm_pred)
            #np.set_printoptions(precision=2)
           #ax = plot_confusion_matrix(cm_label, cm_pred, classes=classes_names,
                      #title='Confusion matrix, without normalization')
   
    #plt.savefig('/mnt/data2/betty/webvision_train/results/resnet50/confusion_matrix_cls_names.png')
        ave_loss = v_value_loss / num_batch
        #ave_loss = ave_loss.detach().numpy()
        ave_acc = (num_correct / float(num_samples))
        ave_top5 = (num_correct5 / float(num_samples))

        if idx % save_freq == (save_freq - 1):
            #np.savetxt("/mnt/data2/betty/webvision_train/results/resnet50/conf_mat_training/all_pred_testing2.csv", all_pred_m, delimiter=",")
            #print('all_pred_m:', all_pred_m.shape )
            time_batch=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('[valid {0:d}] ' 'loss: {1:.4f}\t' 'acc: {2:.4f}\t'  '{batch_time} \t'
                     'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t' 'acc_top5 {top5_avg:.3f}'.format(
                epoch + 1, ave_loss, ave_acc, top5=top5, top5_avg=ave_top5, batch_time=time_batch))

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            # 1. Log scalar values (scalar summary)
            info = { 'loss_val': ave_loss, 'accuracy1_val': ave_acc , 'accuracy5_val' : ave_top5}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, idx)

            # 2. Log values and gradients of the parameters (histogram summary)
        

    return ave_loss, ave_acc, ave_top5


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