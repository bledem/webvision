from __future__ import print_function
import os
import pickle

import numpy
import time
import numpy as np
import torch
from collections import OrderedDict
import pickle as cPickle
from torch.autograd import Variable
import numpy as np
import shutil

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)

def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 5
    index_list = []
    npts=int(npts)
    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] / 5
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])
    npts=int(npts)

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def save_encoding(model, data_loader, opt, save_path, concat):
    model.val_start()
    img_embs = None
    img_emb_list = []
    concat_list =[]
    nb=0
    idx=0
    print("Writing the image names...")
    np.savetxt(os.path.join(save_path, "img_names.lst"), data_loader.dataset.imgs_path, delimiter='\n', fmt="%s")
          
    print("Writing the pickles...")
    for i, data in enumerate(data_loader): #
        images=data[0]
        name= data[1]
        img_path = name[0].strip('\n')[-13:-4]
        #print("name", img_path)

        with torch.no_grad():
            images = Variable(images)
        if torch.cuda.is_available():
            images = images.cuda()
        output = model.img_enc(images)
        if concat:
            concat= [images.data.squeeze().cpu().detach().numpy(), output.squeeze().cpu().detach().numpy()]
            concat_list.append(concat)
            #print("shape of concat is", len(concat_list), concat[0].shape, concat[1].shape)
            with open(os.path.join(save_path, str(img_path)+".pkl"), 'wb') as f:
                cPickle.dump(concat, f, protocol=cPickle.HIGHEST_PROTOCOL) 
        else:
            img_emb_list.append(model.img_enc(images))
            print("encoding", i, data_loader.dataset.imgs_path[i], len(name), images.shape )
    
    if not concat:
        result=torch.cat(img_emb_list)
        print("total lenght is", len(data_loader), len(img_emb_list), result.shape)
        result=result.data.cpu().numpy()

        while (5000*idx<len(result)):
            with open(os.path.join(save_path, "img_val"+str(idx)+".pkl"), 'wb') as f:
                cPickle.dump(result[idx*5000:(idx*5000)+5000], f, protocol=cPickle.HIGHEST_PROTOCOL) 
            nb=nb+5000
            idx=idx+1
    print("we saved for concat", len(concat_list), "val data" )
    



def save_encoding_train(model, data_loader, save_path):
    model.val_start()
    img_embs = None
    img_emb_list = []
    prev_query="q00001"
    next_query=""
    idx_local=0
    with open(os.path.join(save_path, "img_names_all.lst"), 'w') as txt:
        for path in data_loader.dataset.light_dataset:
            #print("path", path[0])
            txt.write(path[0])
    
    for i, (images, captions, lengths, ids, cls_id, imgs_path, query_id, img_id) in enumerate(data_loader): #images, targets, lengths, ids, cls_ids, imgs_path
        #print("ids", imgs_path)

        print("next batch", imgs_path[0])
        img_emb_list.append(imgs_path)
        img_emb, cap_emb = model.forward_emb(images=images, captions=captions, lengths=lengths)
        for i, feat in enumerate(range(0, len(img_emb))):
            if not os.path.exists(os.path.join(save_path, "img_feat", str(query_id[i]))):
                os.makedirs(os.path.join(save_path, "img_feat", str(query_id[i])))
            with open(os.path.join(save_path, "img_feat", str(query_id[i]), str(img_id[i])+".pkl"), 'wb') as f:
                cPickle.dump(img_emb[i].data.cpu().numpy(), f, protocol=cPickle.HIGHEST_PROTOCOL)
            
            
       # print("encoding", i, data_loader.dataset.dataset[0].imgs_path[i], len(names), images.shape )
        #print("encoding", i,  images.shape )
    with open(os.path.join(save_path, "img_names_all.lst"), 'w') as txt:
        for batch_path in img_emb_list:
            for path in batch_path:
                txt.write(path)
    # for folder in os.listdir(os.path.join(save_path, "img_feat")):
    #     print("Making archive")
    #     shutil.make_archive(os.path.join(save_path, "img_feat", folder), 'zip', os.path.join(save_path, "img_feat", folder))
    #     print("Deleting archive")
    #     shutil.rmtree(os.path.join(save_path, "img_feat", folder))
                   
    #result=torch.cat(img_emb_list)
    #print("total lenght is", len(data_loader), len(img_emb_list), result.shape)
    #result=result.data.cpu().numpy()
    #print("we saved ", len(result), "val data" )
    # with open(os.path.join(save_path, "img_names.lst"), 'w') as f:
    #     for img_path in names:
    #         f.write(img_path)
    # while (5000*idx<len(result)):
    #     with open(os.path.join(save_path, "img_train"+str(idx)+".pkl"), 'wb') as f:
    #         cPickle.dump(result[idx*5000:(idx*5000)+5000], f, protocol=cPickle.HIGHEST_PROTOCOL) 
    #     nb=nb+5000
    #     idx=idx+1


def save_img_path_train(data_loader, save_path):

    for i, output in enumerate(data_loader): #
        images=output[0]
        ids = output[5]
        print("ids", ids[0])
        with open(os.path.join(save_path, "img_names_train_backup.lst"), 'a') as f:
            for path in ids:
                f.write(path.decode('utf8'))

    # with open(os.path.join(save_path, "img_names_train.lst"), 'w') as f:
    #     for img_path in names:
    #         for path in img_path:
    #             f.write(path.decode('utf8'))





def encode_data(model, data_loader, log_step=1, rand=True, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """

    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None
    for i, (images, captions, lengths, ids, cls_id, img_id) in enumerate(data_loader): #images, targets, lengths, ids, cls_ids, img_id
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb = model.forward_emb(images, captions, lengths,
                                             volatile=True)

        # initialize the numpy arrays given the size of the embeddings
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

        # preserve the embeddings by copying from gpu and converting to numpy
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids] = cap_emb.data.cpu().numpy().copy()

        #print("in encode_data, img emb are",img_emb," \n cap emb", cap_emb)

        # measure accuracy and record loss
        loss = model.forward_loss(img_emb, cap_emb)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss:.3f}'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger), loss=loss))
        del images, captions
    return img_embs, cap_embs


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