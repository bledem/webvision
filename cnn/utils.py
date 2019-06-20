# -*- coding: utf-8 -*-
# Author: Xiaoming Qin

"""Utility function for train(test) a neural network"""

import torch
import shutil
import os
from os.path import join
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels




def save_ckpt(state, arch, is_best=False):
    fname = join('../results', arch, 'ckpt.tar')

    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, join('../results', arch, 'model_best.tar'))





def adjust_lr(optimizer, epoch, lr, dec_freq=3):
    """Sets the learning rate to the init `lr`
       decayed by 10 every `dec_freq` epochs
    """
    new_lr = lr * (0.1 ** (epoch // dec_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def adjust_lr_manual(optimizer, epoch, lr_epoch_map):
    """ Set the learning rate by the `lr` which is
        defined by user
    """
    if epoch in lr_epoch_map.keys():
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_epoch_map[epoch]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def plot_confusion_matrix_2(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    np.savetxt("/mnt/data2/betty/webvision_train/results/resnet50/conf_mat_training/confusion_matrix.csv", cm, delimiter=",")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # y_true = np.array(y_true.ravel().astype(int))
    # y_pred = np.array(y_pred.ravel().astype(int))

    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    

    #print(cm)
    np.savetxt("/mnt/data2/betty/webvision_train/results/resnet50/conf_mat_training/confusion_matrix.csv", cm, delimiter=",")

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_history(fname):
    import matplotlib.pyplot as plt

    with open(fname, 'rb') as f:
        infos = f.readlines()

    model_name = infos[0].strip('\n').split(": ")[1]
    batch_size = infos[1].strip('\n').split(": ")[1]
    epochs = infos[2].strip('\n').split(": ")[1]
    best_epoch = infos[3].strip('\n').split(": ")[1]
    save_name = fname.split('/')[-1].split('.')[0]

    train_loss = []
    train_acc = []
    train_steps = []

    ln_i = 5
    while infos[ln_i].startswith('valid') is False:
        _, step, loss, acc = infos[ln_i].strip('\n').split(' ')
        train_loss.append(float(loss))
        train_acc.append(float(acc))
        train_steps.append(int(step))
        ln_i = ln_i + 1

    valid_loss = []
    valid_acc = []
    valid_epoch = []

    for i in xrange(ln_i + 1, len(infos)):
        epoch, loss, acc = infos[i].strip('\n').split(' ')
        valid_loss.append(float(loss))
        valid_acc.append(float(acc))
        valid_epoch.append(int(epoch))

    record_per_epoch = len(train_steps) / max(valid_epoch)

    for i in xrange(len(train_steps)):
        train_steps[i] = record_per_epoch * train_steps[0] * \
            (i // record_per_epoch) + train_steps[i]

    valid_epoch = [e * record_per_epoch * train_steps[0] for e in valid_epoch]

    plt.figure(1)

    plt.plot(train_steps, train_loss, 'r-',
             label='train loss',
             lw=1.2)
    plt.plot(valid_epoch, valid_loss, 'c-',
             label='valid loss',
             lw=1.2)

    plt.xlabel('training steps')
    plt.ylabel('loss')
    plt.title(model_name + " loss")
    plt.legend(loc='upper right')
    plt.savefig('../results/' + model_name +
                "/loss_{}.png".format(
                    save_name))

    plt.figure(2)
    plt.plot(train_steps, train_acc, 'r-',
             label='train acc',
             lw=1.2)
    plt.plot(valid_epoch, valid_acc, 'c-',
             label='valid acc',
             lw=1.2)

    plt.xlabel('training steps')
    plt.ylabel('accuracy')
    plt.title(model_name + " accuracy")
    plt.legend(loc='upper left')
    plt.savefig('../results/' + model_name +
                "/acc_{}.png".format(
                    save_name))

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)) #0 if no similar 1 if same number

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def name_of_class(path="/mnt/data2/betty/Pictures/webvision/info/synsets.txt"):
    classes_names = []
    synset_name = []
    with open(path) as f:
        for line in f:
            name = ""
            synset_name.append( line.split()[0][1:])
            for i in range(1,len(line.split())):
                name = name + "/"+ line.split()[i]      
            classes_names.append(line.split()[1])
    #print(classes_names, synset_name)

    return classes_names, synset_name