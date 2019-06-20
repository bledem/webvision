import pickle
import os
import time
import shutil
import torch
import gensim
import gensim.downloader as api
from gensim.test.utils import datapath
#import data
#from vocab import Vocabulary  # NOQA
#from cnn.encoder import VSE
#from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data
import numpy as np
import logging
import tensorboard_logger as tb_logger
from datasets.folder import EmbeddedFolder, collate_fn 
import argparse
from cnn.encoder import VSE
from tool.embedding_tool import AverageMeter, LogCollector


def encode_data(model, data_loader, log_step=1, logging=print):
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
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
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
        print("in encode data", i "step", )
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
	parser.add_argument('--batch_size', default=128, type=int,
						 help='Size of a training mini-batch.')
	parser.add_argument('--word_dim', default=300, type=int,
						 help='Dimensionality of the word embedding.')
	parser.add_argument('--embed_size', default=1024, type=int,
						 help='Dimensionality of the joint embedding.')
	parser.add_argument('--grad_clip', default=2., type=float,
						 help='Gradient clipping threshold.')
	parser.add_argument('--crop_size', default=224, type=int,
						 help='Size of an image crop as the CNN input.')
	parser.add_argument('--num_layers', default=1, type=int,
						 help='Number of GRU layers.')
	parser.add_argument('--learning_rate', default=.0002, type=float,
						 help='Initial learning rate.')
	parser.add_argument('--lr_update', default=15, type=int,
						 help='Number of epochs to update the learning rate.')
	parser.add_argument('--workers', default=4, type=int,
						 help='Number of data loader workers.')
	parser.add_argument('--log_step', default=10, type=int,
						 help='Number of steps to print and record the log.')
	parser.add_argument('--val_step', default=500, type=int,
						 help='Number of steps to run validation.')
	parser.add_argument('--logger_name', default='runs/runX',
						 help='Path to save the model and Tensorboard log.')
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
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
	parser.add_argument('--root_img', default='/mnt/data3/web_feature_training/google',
						 help='img or feature root folder')
	parser.add_argument('--root_json', default='/mnt/data1/webvision2.0/google/',
						 help='json root folder')

	opt = parser.parse_args()
	print(opt)

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
	#w2v_model = api.load("wiki-news-300d-1M-subword")
	vocab = w2v_model.wv
	#word2idx = dict([(k, v.index) for k, v in w2v_model.vocab.items()])
	#idx2word = dict([(v, k) for k, v in w2v_model.vocab.items()])
	
	opt.vocab_size = len(vocab.vocab)
	print("vocab is", opt.vocab_size)
	val_dataset=  EmbeddedFolder(opt.root_json, opt.root_img, w2v_model, dict_=query_synset_dic)
	val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size= opt.batch_size, pin_memory=True, collate_fn=collate_fn)

	# for i in range(len(val_loader)):
	# 	img_feat, caption, target, id = val_dataset[i]
		#print(caption, id)
	model = VSE(opt, w2v_model)
	encode_data(model, val_loader, log_step=10, logging=print)

if __name__ == '__main__':


	main()