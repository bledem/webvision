import torch
import gensim
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from cnn.factory import get_model
import torch.optim as optim
from cnn.factory import get_model




def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X
def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='resnet', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('precomp'):
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, use_abs, no_imgnorm)
    else:
        img_enc = EncoderImageFull(
            embed_size, finetune, cnn_type, use_abs, no_imgnorm)

    return img_enc


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size=256, finetune=False, cnn_type='resnet50',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        model = get_model(name=cnn_type, num_classes=5607)
        model = torch.nn.DataParallel(model)
        model.to("cuda")
        checkpoint = torch.load("/mnt/data2/betty/webvision_train/results/resnet50/5000classes_onemonth/model_best.tar")
        model.load_state_dict(checkpoint['state_dict'])
        
        print("Successfully load the saved model at model_best.tar") 

        self.cnn = model


        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer of CNN with a new one
      
        if cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()
        else:
            print("error in chosing the architecture")
            return

        self.init_weights()

    def load_state_dict(self, load_path):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)
        

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim=2048, embed_size=51, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size #
        self.no_imgnorm = no_imgnorm #normalize image
        self.use_abs = use_abs
#define the layers
        self.fc = nn.Linear(img_dim, embed_size)
#define the init function
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are not already l2-normalized
        images = l2norm(images.view( images.size(0), -1))
        #print(images.shape, self.fc )
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class PretrainedEncoderText(nn.Module):
    """Creating embedding vectors which takes as input integers, 
    it looks up these integers into an internal dictionary, and it returns the associated vectors."""
    def __init__(self, vocab_path, w2v_model, gpu_id=0,  num_layers=2, vocab_size=256 ,embed_size=256,
                 use_abs=False):
        super(PretrainedEncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size
        self.gpu_id = gpu_id

        weights= torch.FloatTensor(w2v_model.vectors)
        #self.embed = nn.Embedding.from_pretrained(weights) #size is vocab_size, hidden_size
        
        num_embeddings, embedding_dim = weights.size()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.embed.load_state_dict({'weight': weights})
        self.embed.weight.requires_grad = False
        #print("embedding layer", self.embed)

        # caption embedding
        self.rnn = nn.GRU(embedding_dim, self.embed_size, num_layers, batch_first=True)
#maybe linear here

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x.long())
        #print("in forward, lenght", lengths)
        packed = pack_padded_sequence(x, lengths, batch_first=True) 
        ## packed_input.data.shape : (batch_sum_seq_len X embedding_dim) 

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda(self.gpu_id)
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, word_dim,  num_layers=2, vocab_size=256 ,embed_size=256,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

        self.init_weights()
        # if torch.cuda.is_available():
        #     self.embed.cuda()
        #     self.rnn.cuda()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)


    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True) 
        ## packed_input.data.shape : (batch_sum_seq_len X embedding_dim) 

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t()) #image.mm(sentence.t()) & mm() Performs a matrix multiplication of the matrices 


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, gpu_id=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.gpu_id=gpu_id
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1) #we take the diag of the multiplication result
        d1 = diagonal.expand_as(scores) #Expands this tensor to the size of the specified tensor.
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda(self.gpu_id)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class VSE(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt, w2v_model, gpu_id=0):  
        # tutorials/09 - Image Captioning
        #opt is the model loaded as a checkpoint in eval or train arg parser
        # Build Models

        self.grad_clip = opt.grad_clip #Gradient clipping threshold default=2
        self.img_enc = EncoderImagePrecomp( img_dim=2048, embed_size=256, use_abs=False, no_imgnorm=False)
        # self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
        #                            opt.embed_size, opt.num_layers,
        #                            use_abs=opt.use_abs)
        #self.img_enc = EncoderImageFull()
        self.txt_enc = PretrainedEncoderText(opt.vocab_path, w2v_model, num_layers=2, vocab_size=256 ,embed_size=256,
                 use_abs=False)
        self.img_enc.eval()
        self.txt_enc.eval()
        self.gpu_id = gpu_id

        if torch.cuda.is_available():
            self.img_enc.cuda(gpu_id)
            self.txt_enc.cuda(gpu_id)
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.module.img_enc.train()
        self.module.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        with torch.no_grad():
            images = Variable(images)
            captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda(self.gpu_id)
            captions = captions.cuda(self.gpu_id)

        # Forward
        img_emb = self.img_enc(images) #shape of both embed is [49, 256]
        cap_emb = self.txt_enc(captions, lengths) #call the forward
        #print("shape of emb", img_emb.shape, cap_emb.shape)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data, img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
        #print("loss", loss, "img_emb", img_emb.shape, cap_emb.shape)
        return loss, img_emb, cap_emb
def __getattr__(self, name):
        return getattr(self.module, name)








class FusionNet(nn.Module):
    """
    take the feature vector after resnet baseline and after fc trained with triplet loss 
    to train 2 FC layers
    """
    def __init__(self, opt, img_dim=2304, embed_size=5000, use_abs=False, no_imgnorm=False):
        super(FusionNet, self).__init__()
        self.embed_size = embed_size #
        self.no_imgnorm = no_imgnorm #normalize image
        self.use_abs = use_abs
#define the layers
        
        self.fc1 = nn.Linear(img_dim, embed_size)
        self.relu =  nn.ReLU()
        self.fc2 = nn.Linear(embed_size, embed_size)
        #self.softmax= nn.LogSoftmax(dim=1)

#define the init function
        self.init_weights()
         

        self.Eiters = 0

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc1.in_features +
                                  self.fc1.out_features)
        self.fc1.weight.data.uniform_(-r, r)
        self.fc1.bias.data.fill_(0)
        r = np.sqrt(6.) / np.sqrt(self.fc2.in_features +
                                  self.fc2.out_features)
        self.fc2.weight.data.uniform_(-r, r)
        self.fc2.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are not already l2-normalized
        x = l2norm(images.view( images.size(0), -1))
        #print(images.shape, self.fc )
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        #x = F.log_softmax(x) #no need of log softmax here if we use cross entropy as loss
        #x = self.softmax(x)
        # normalize in the joint embedding space
        

        return x

class FusionNet_three(nn.Module):
    """
    take the feature vector after resnet baseline and after fc trained with triplet loss 
    to train 2 FC layers
    """
    def __init__(self):
        super(FusionNet_three, self).__init__()
        
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 5000)
        self.relu =  nn.ReLU()

#define the init function
        self.init_weights()
         

        self.Eiters = 0

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """

        r = np.sqrt(1.) / np.sqrt(self.fc3.in_features +
                                  self.fc3.out_features)
        self.fc3.weight.data.uniform_(-r, r)
        self.fc3.bias.data.fill_(0)
        r = np.sqrt(1.) / np.sqrt(self.fc2.in_features +
                                  self.fc2.out_features)
        self.fc2.weight.data.uniform_(-r, r)
        self.fc2.bias.data.fill_(0)
        r = np.sqrt(1.) / np.sqrt(self.fc1.in_features +
                                  self.fc1.out_features)
        self.fc1.weight.data.uniform_(-r, r)
        self.fc1.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are not already l2-normalized
        #x = l2norm(images.view( images.size(0), -1))
        #print(images.shape, self.fc )
        resnet_feat=torch.empty(len(images),2048)
        vse_feat=torch.empty(len(images),256)

        for idx, feat_concat in enumerate(images):
            #print("check", feat_concat[:2048].shape, feat_concat[2048:].shape)
            #resnet_feat[idx,:] = feat_concat[:2048]
            vse_feat[idx,:] = feat_concat[2048:]
        x = self.relu(self.fc1(vse_feat.cuda()))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ScaleLayer(nn.Module):

   def __init__(self, init_value=1e-3):
       super().__init__()
       self.scale = nn.Parameter(torch.FloatTensor([init_value]))

   def forward(self, input):
       return input * self.scale


class FusionNet_scale(nn.Module):
    """
    take the feature vector after resnet baseline and after fc trained with triplet loss 
    to train 2 FC layers
    """
    def __init__(self, opt, img_dim=2048, embed_size=5607, use_abs=False, no_imgnorm=False):
        super(FusionNet_scale, self).__init__()
        self.embed_size = embed_size #
        self.no_imgnorm = no_imgnorm #normalize image
        self.use_abs = use_abs
        self.resnet_weight = opt.params_file
#define the layers
        
        self.fc1 = nn.Linear(img_dim, embed_size)
        self.relu =  nn.ReLU()
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 5000)
        self.scale = ScaleLayer(0)

        #self.softmax= nn.LogSoftmax(dim=1)

#define the init function
        self.init_weights()
         

        self.Eiters = 0

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """

        params = torch.load(self.resnet_weight)

        self.fc1.weight.data = params['state_dict']['module.fc.weight'].clone()
        self.fc1.bias.data = params['state_dict']['module.fc.bias'].clone()


        r = np.sqrt(1.) / np.sqrt(self.fc3.in_features +
                                  self.fc3.out_features)
        self.fc3.weight.data.uniform_(-r, r)
        self.fc3.bias.data.fill_(0)
        r = np.sqrt(1.) / np.sqrt(self.fc2.in_features +
                                  self.fc2.out_features)
        self.fc2.weight.data.uniform_(-r, r)
        self.fc2.bias.data.fill_(0)
        r = np.sqrt(1.) / np.sqrt(self.fc4.in_features +
                                  self.fc4.out_features)
        self.fc4.weight.data.uniform_(-r, r)
        self.fc4.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are not already l2-normalized
        #x = l2norm(images.view( images.size(0), -1))
        #print(images.shape, self.fc )
        resnet_feat=torch.empty(len(images),2048)
        vse_feat=torch.empty(len(images),256)

        for idx, feat_concat in enumerate(images):
            #print("check", feat_concat[:2048].shape, feat_concat[2048:].shape)
            resnet_feat[idx,:] = feat_concat[:2048]
            vse_feat[idx,:] = feat_concat[2048:]
        x1 = self.fc1(resnet_feat.cuda())

        x2 = self.relu(self.fc2(vse_feat.cuda()))
        x2 = self.relu(self.fc3(x2))
        x2 = self.scale(self.fc4(x2))
        #print(x2.shape, x1.shape, x1[:,:5000].shape)
        x = x1[:,:5000]+x2
        

        return x
