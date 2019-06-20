# -*- coding: utf-8 -*-
# Author: Xiaoming　Qin

"""Train a convolutional neural network"""

import sys
import os
import torch
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt 
import torch.nn as nn
from PIL import Image
from os.path import join as pjoin
from torch.autograd import Variable
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets.folder import GenImageFolder, SpecImageFolder, DriveData, SampleData
from cnn.utils import adjust_lr, save_ckpt, adjust_lr_manual, plot_confusion_matrix, accuracy, AverageMeter, name_of_class, plot_confusion_matrix_2
from cnn import utils
from cnn.factory import get_model
from cnn.config import cfg
from cnn.logger import Logger
from sklearn.metrics import confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] =  "1,3" #we give this number as the device id we use

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images_so_far = 0


def parse_args():

    parser = argparse.ArgumentParser(description='Train a deep neural network')
    parser.add_argument('--data-path', help='path to data root',
                        default='/data/wv-40', type=str)
    parser.add_argument('--gpus', help='GPU id to use',
                        default='0', type=str)
    parser.add_argument('--epochs', help='number of epochs',
                        type=int, default='90')
    parser.add_argument('--batch-size', help='mini-batch size',
                        default=16, type=int)
    parser.add_argument('--lr', help='initial learning rate',
                        default=0.1, type=float)
    parser.add_argument('--weight-decay', help='learing weight decay',
                        default=1e-4, type=float)
    parser.add_argument('--num-workers', help='number of workers',
                        default=4, type=int)
    parser.add_argument('--arch', dest='model_name',
                        help='model to train on',
                        default='inception_v3', type=str)
    parser.add_argument('--input-size', help='size of model input',  #all other than inception reauires 224 , inveptionv3 299
                        default=299, type=int)
    parser.add_argument('--print-freq', help='print frequency',
                        default=100, type=int)
    parser.add_argument('--data-folder', help='choose folder google, flickr, both',
                        default='both', type=str)
    parser.add_argument('--save-freq', help='iteration frequency',
                        default=20000, type=int)
    parser.add_argument('--load-path', help='checkpoint path',
                        default='None', type=str)
    parser.add_argument('--query', help='number of ImagetNet synset out of 5000',
                        default='None', type=int)
    parser.add_argument('--class-weighted', help='adding class weighted system for improving ',
                        default='False', type=bool)



    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def validate(valid_loader, model, criterion, epoch, log, print_freq, save_freq):
    print("Start validation..")
    classes_names, synset_name = name_of_class()

    # switch to evaluate mode
    model.eval()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    end = time.time()

    num_correct = 0.
    num_samples = 0.
    num_batch = 0
    v_value_loss = 0.
    num_correct5 = 0.

#for the confusion matrix
    cm_label=[];
    cm_pred=[];
    all_pred_m= np.array([], dtype=np.int64).reshape(0,5000)

    

    for idx, data in enumerate(valid_loader, 0):
        num_batch = idx + 1
        inputs, labels = data       
        #cm_label= labels.cpu().numpy()

        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device)) #  labels = Variable(labels.to(device)).cuda() to force on device 3

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
        cm_label = np.concatenate([cm_label,labels.data.cpu().numpy()])
        cm_pred = np.concatenate([cm_pred, preds.data.cpu().numpy().astype(int)])
        #np.set_printoptions(suppress=True)
        #for top5 acc
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))

        #for kato-san images-all_pred inference
        #for i in range(ps.shape[0]):
        all_pred_m= np.vstack((all_pred_m, output_soft.data.cpu().numpy()[:,:5000]))

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

        break
        if idx % print_freq == (print_freq - 1):
            print( 'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t' 
                 'Acc@5 {top5_val:.3f} ({top5.avg:.3f})\t' .format(top1_val=top1.val/float(labels.size(0)),
                  top5_val=correct_5.item()/float(labels.size(0)),
                  top1=top1, top5=top5))
            print('soft shape:', output_soft.shape, " ,top value of soft:", value_soft, " ,top index:", preds_soft ," ,gt:", labels)

            #cnf_matrix = confusion_matrix(cm_label, cm_pred)
            #np.set_printoptions(precision=2)
           #ax = plot_confusion_matrix(cm_label, cm_pred, classes=classes_names,
                      #title='Confusion matrix, without normalization')
   
    #plt.savefig('/mnt/data2/betty/webvision_train/results/resnet50/confusion_matrix_cls_names.png')
        if idx % save_freq == (save_freq - 1):
            np.savetxt("/mnt/data2/betty/webvision_train/results/resnet50/conf_mat_training/all_pred_testing2.csv", all_pred_m, delimiter=",")
            print('all_pred_m:', all_pred_m.shape )



    #np.savetxt("/mnt/data2/betty/webvision_train/results/resnet50/conf_mat_training/all_pred_testing2.csv", all_pred_m, delimiter=",")
    ave_loss = v_value_loss / num_batch
    ave_acc = num_correct / float(num_samples)
    ave_top5 = num_correct5 / float(num_samples)

    #log.record_val([epoch + 1, ave_loss, ave_acc, ave_top5], epoch)

    #Confusion matrix plotting
    # cnf_matrix = confusion_matrix(cm_label, cm_pred)
    # np.set_printoptions(precision=2)
    # ax = plot_confusion_matrix(cm_label, cm_pred, classes=np.arange(5000),
    #                   title='Confusion matrix, without normalization')
    # plt.savefig('/mnt/data2/betty/webvision_train/results/resnet50/confusion_matrix.png')
    # ax = plot_confusion_matrix(cm_label, cm_pred, classes=classes_names,
    #                   title='Confusion matrix, without normalization')
    # plt.savefig('/mnt/data2/betty/webvision_train/results/resnet50/confusion_matrix_cls_names.png')

    # plt.show()




    print('[valid {0:d}] ' 'loss: {1:.4f}\t' 'acc: {2:.4f}\t'  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
             'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t' 'acc_top5 {top5_avg:.3f}'.format(
        epoch + 1, ave_loss, ave_acc, top5=top5, top5_avg=ave_top5, batch_time=time.time() - end))
    print('[valid {0:d}] ' 'loss: {1:.4f}\t' 'acc: {2:.4f}\t' 'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
             'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t' 'acc_top5 {top5_avg:.3f}'.format(
        epoch + 1, ave_loss, ave_acc,top1=top1, top5=top5, top5_avg=ave_top5, batch_time=time.time() - end))

def test(train_loader, model, print_freq):
    top1 = AverageMeter()
    top5 = AverageMeter()
    cm_label=[];
    cm_pred=[];
    model.eval()
    num_correct = 0.
    num_samples = 0.
    classes_names, synset_name = name_of_class()

    for idx, data in enumerate(train_loader, 0):
        inputs, labels = data

        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
 
        # feedforward
        output = model(inputs)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        _, preds = torch.max(output, 1)
        correct = (preds == labels).sum()
        # print('Predicted: ', ' //'.join('%5s' % preds[j].data.cpu().numpy() for j in range(len(labels))))
        # print('Ground Truth is: ', ' //'.join('%5s' % labels[0].data.cpu().numpy() for j in range(len(labels))))
        num_samples += labels.size(0)
        num_correct += correct.item()

        cm_label = np.concatenate([cm_label,labels.data.cpu().numpy()])
        cm_pred = np.concatenate([cm_pred, preds.data.cpu().numpy().astype(int)])

        if idx % print_freq == (print_freq - 1):
            running_acc = num_correct / float(num_samples)

            #print('Predicted: ', ' //'.join('%5s' % preds[0].data.cpu().numpy() ))
            #print('Ground Truth is: ', ' //'.join('%5s' % labels[0].data.cpu().numpy() ))

            print(   'acc: {0:.4f}' 'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t' 'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                 running_acc, 
                    top1=top1, top5=top5 ))

            num_samples, num_correct = 0., 0.
            cm = confusion_matrix(cm_label, cm_pred)
            print(cm.shape)
            np.savetxt("/mnt/data2/betty/webvision_train/results/resnet50/conf_mat_training/confusion_matrix_training2.csv", cm, delimiter=",")


    cnf_matrix = _usion_matrix(cm_label, cm_pred)
    #np.set_printoptions(precision=2)
    ax = plot_confusion_matrix(cm_label, cm_pred, classes=np.arange(5000),
                      title='Confusion matrix, without normalization')
    #plt.savefig('/mnt/data2/webvision_train/results/resnet50/conf_mat_training/confusion_matrix.png')
    ax = plot_confusion_matrix(cm_label, cm_pred, classes=classes_names,
                      title='Confusion matrix, without normalization')
    #plt.savefig('/mnt/data2/betty/webvision_train/results/resnet50/conf_mat_training/confusion_matrix_cls_names.png')

def train(train_loader, model, criterion,
          optimizer, epoch, log, print_freq, save_freq, model_name):
    global images_so_far
    # switch to train mode
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    print("[epoch: {}]".format(epoch + 1))

    running_loss, running_acc = 0., 0.
    num_samples, num_correct = 0., 0.
    info = []

    for idx, data in enumerate(train_loader, 0):
        data_time.update(time.time() - end)
        inputs, labels = data
        images_so_far += len(labels)

        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
 
        optimizer.zero_grad()

        # feedforward
        output = model(inputs)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        _, preds = torch.max(output, 1)
        correct = (preds == labels).sum()

        num_samples += labels.size(0)
        num_correct += correct.item()
        running_loss += loss.item()

        if idx % print_freq == (print_freq - 1):
            running_acc = num_correct / float(num_samples)
            ave_running_loss = running_loss / print_freq

            print('[epoch {0:d} ' 'nb done batches  {1:4d}\t' 'seen:{4:d}\t' 'loss: {2:.4f}\t' ' acc: {3:.4f}'
             'Batch_time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch + 1, idx + 1,
                ave_running_loss,
                running_acc, images_so_far, batch_time=batch_time,
                    top1=top1, top5=top5 ))
        
            log.record_trn([epoch + 1, idx + 1,
                            ave_running_loss,
                            running_acc])

            num_samples, num_correct = 0., 0.
            running_loss = 0.


        if idx % save_freq == (save_freq - 1):
            print("saving at", idx, "iteration")
            torch.save({'arch': model_name,
                       'state_dict': model.state_dict(), #add .module because of DataParallel 
                       'epoch' : epoch,
                       'idx' : idx,
                       'optimizer_state_dict': optimizer.state_dict()},
                      os.path.join('../results', model_name, 'in_between.tar'))
            log.save()



def train_model(data_root, gpus, epochs, data_folder="dummy",
                batch_size=64, base_lr=0.1,
                model_name='alexnet',
                weight_decay=0.,
                num_workers=1,
                in_size=224,
                num_classes=5607,  #number of query+1 I guess
                print_freq=100,
                save_freq=20000,
                load_path='/mnt/data2/betty/webvision_train/results/alexnet/first_result', queries={}, query_synset_dic_={},
                class_weighted_= False):


    #normalize = transforms.Normalize(mean=cfg.PIXEL_MEANS,
     #                                std=cfg.PIXEL_STDS)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Image transformer
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(in_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(in_size),
            transforms.ToTensor(),
            normalize])


    valid_data = DriveData(folder_dataset=data_root, dataset_type='val',  #val_images_resized
                                transform=valid_transform, queries_= queries, dict_=query_synset_dic_)
    print("flickr")
    train_data_flickr = SampleData(root=data_root, folder='flickr_images_resized', #reduced alias of google 10 webvision in Picture folder
                                transform=train_transform, queries_= queries, dict_=query_synset_dic_)

    print("google")
    train_data_google = SampleData(root=data_root, folder='google_images_resized', 
                                transform=train_transform, queries_= queries, dict_=query_synset_dic_)

    print("Validation and training data loaded")
    #train_data_dummy = GenImageFolder(root=pjoin(data_root, 'val_images_resized'), #val
    #                            transform=train_transform)
    
    #train_loader = DataLoader(dataset=train_data_dummy, batch_size=batch_size,
    #                         shuffle=True, num_workers=num_workers,
    #                        pin_memory=True) 
    if(data_folder=='google'):
        train_loader = DataLoader(dataset=train_data_google, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
        print("training from google, on" , len(train_data_google))
        class_weight_google = train_data_google.class_fq

    elif (data_folder=='flickr'):
        train_loader = DataLoader(dataset=train_data_flickr, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
        print("training from flickr on" , len(train_data_flickr) )
        class_weight_flickr = train_data_flickr.class_fq

    elif (data_folder=='both'):
        concat_data = ConcatDataset((train_data_google, train_data_flickr))
        print("training from both, on" , len(train_data_google)+len(train_data_flickr) )
        #class_weight = class_weight_google+ class_weight_flickr
        train_loader = DataLoader(dataset= concat_data, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    #else:
     	#print("training from dummy, on" , len(train_data_dummy) )


    
    valid_loader = DataLoader(dataset=valid_data, batch_size=4,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    # define the model
    model = get_model(name=model_name, num_classes=num_classes)

    if len(gpus) > 1:
        # prev_gpus = gpus
        # gpus = [int(i) for i in gpus.strip('[]').split(',')]
        # print("Let's use", len(gpus), "on", torch.cuda.device_count(), "GPUs!")
        # os.environ["CUDA_VISIBLE_DEVICES"] =  prev_gpus #we give this number as the device id we use
        # gpus_idx = range(0,len(gpus))
        model = torch.nn.DataParallel(model)
        model.to(device)


    elif len(gpus)==1:
        print("one gpu: ", gpus)
        prev_gpus = gpus
        gpus = [int(i) for i in gpus.strip('[]').split(',')]
        print("Let's use", gpus, "on", torch.cuda.device_count(), "GPUs!")
        os.environ["CUDA_VISIBLE_DEVICES"] =  prev_gpus #we give this number as the device id we use
        gpus_idx = range(0,len(gpus))
        checkpoint = torch.load(os.path.join(load_path, "one_gpu_best.tar"))
        model.load_state_dict(checkpoint['state_dict'])
        print("Successfully load the saved model at",os.path.join(load_path, "one_gpu_best.tar") )
        #model.cuda(3) #to force
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print("no gpus")

    feature_extract = False
    params_to_update = model.parameters()

    ##un comment to PRINT the model
    #print("Params to learn:")
    # if feature_extract:
    #     params_to_update = []
    #     for name,param in model.named_parameters():
    #         if param.requires_grad == True:
    #             params_to_update.append(param)
    #             print("\t",name)
    # else:
    #     for name,param in model.named_parameters():
    #         if param.requires_grad == True:
    #             print("\t",name)


    # define optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                weight_decay=weight_decay,
                                momentum=0.9)
    if (class_weighted_):
        print("use class weighted strategy")
        loss_func = torch.nn.CrossEntropyLoss(class_weight).to(device)
    else:
        loss_func = torch.nn.CrossEntropyLoss().to(device)

    # define logs
    logger = Logger(arch=model_name, epochs=epochs,
                    batch_size=batch_size)
    # create model results folder
    res_save_fld = os.path.join('../results', model_name)
    if not os.path.exists(res_save_fld):
        os.mkdir(res_save_fld)
    log_save_path = os.path.join('../logs', model_name)
    if not os.path.exists(log_save_path):
        os.mkdir(log_save_path)

    #lr_epoch_map = {0: 0.01, 20: 0.001, 60: 0.0001}
    if (os.path.exists(load_path) and len(gpus)>1):
        checkpoint = torch.load(os.path.join(load_path, "model_best.tar"))
        model.load_state_dict(checkpoint['state_dict'])
        print("Successfully load the saved model at",os.path.join(load_path, "model_best.tar") )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    	#loss_saved = checkpoint['loss']

    for epoch in range(epochs):
    	
        #if (epoch==1 and (os.path.exists(load_path))):
            #epoch = checkpoint['epoch']
            #loss_saved = checkpoint['loss']
    	#adjust_lr_manual(optimizer, epoch, lr_epoch_map)
        validate(valid_loader, model, loss_func, epoch, logger, print_freq, save_freq)

        return
        test(train_loader, model, print_freq)

        adjust_lr(optimizer, epoch, base_lr)
        # train for one epoch

        

        train(train_loader, model, loss_func, optimizer,
              epoch, logger, print_freq, save_freq, model_name)



        


		# Returns the current GPU memory usage by 
		# tensors in bytes for a given device
		#torch.cuda.memory_allocated()
		# Returns the current GPU memory managed by the
		# caching allocator in bytes for a given device
		#torch.cuda.memory_cached()

        # evaluate on validation set
        
       #two models are saved 
        if epoch == logger.best_epoch:
            save_ckpt({'arch': model_name,
                       'state_dict': model.state_dict(),
                       'epoch' : epoch,
                       'optimizer_state_dict': optimizer.state_dict()},
                      model_name, is_best=True)
        else:
            save_ckpt({'arch': model_name,
                   'state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict()},
                  model_name, is_best=False)

        if epoch == logger.best_epoch:
            torch.save({'arch': model_name,
                       'state_dict': model.module.state_dict(), #add .module because of DataParallel for one GPU reading
                        'epoch' : epoch,
                       'optimizer_state_dict': optimizer.state_dict()},
                      os.path.join('../results', model_name, 'one_gpu_best.tar'))
        else :
            torch.save({'arch': model_name,
                       'state_dict': model.module.state_dict(), #add .module because of DataParallel 
                        'epoch' : epoch,
                       'optimizer_state_dict': optimizer.state_dict()},
                      os.path.join('../results', model_name, 'one_gpu.tar'))

    print("Finished Training")
    
    logger.save()


def main():
    args = parse_args()
    queries=[]
    previous=-1
    all=True

    if (args.query is None):
        all=True
    with open(os.path.join(args.data_path, 'info', 'queries_synsets_map.txt')) as f:
            for line in f:
                # Image path
                if(((line.split()[1] != previous) and (len(queries)<=args.query)) or all):
                    queries.append(int(line.split()[0]))
                    previous=line.split()[1]
                    #print(line.split()[0], " ", line.split()[1])
                elif ((len(queries)>=args.query) and not all):
                    #print("over", args.query)
                    break
    query_synset_dic = {}
    with open("/mnt/data2/betty/Pictures/webvision/info/queries_synsets_map.txt") as f:
        for line in f:
           (key, val) = line.split()
           query_synset_dic[int(key)] = val
    #print("We use ", query_synset_dic, "queries as synset map")
    # print("class weighted", args.class_weighted )
    train_model(data_root=args.data_path,
    			data_folder = args.data_folder,
                gpus=args.gpus,
                epochs=args.epochs,
                batch_size=args.batch_size,
                base_lr=args.lr,
                model_name=args.model_name,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                in_size=args.input_size,
                print_freq=args.print_freq,
                save_freq= args.save_freq,
                load_path=args.load_path, 
                queries=queries,
                query_synset_dic_=query_synset_dic,
                class_weighted_= False
                 )

    


if __name__ == "__main__":
    main()
