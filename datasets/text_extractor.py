from torchtext.data import Field
from torchtext.data import TabularDataset
import nltk
import gensim
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models.wrappers import FastText
from textblob import Word
import pandas as pd
import json
import os
from collections import Counter

#nltk.download('stopwords')

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def extract_feat(file_name):
	tokenize = lambda x: x.split()
	TEXT = Field(sequential=True, tokenize=tokenize)
	 
	LABEL = Field(sequential=False, use_vocab=False)


	# tv_datafields = [("id", None), # we won't be needing the id, so we pass in None as the field
	#                  ("comment_text", TEXT), ("toxic", LABEL),
	#                  ("severe_toxic", LABEL), ("threat", LABEL),
	#                  ("obscene", LABEL), ("insult", LABEL),
	#                  ("identity_hate", LABEL)]
	print("file name", file_name)
	tst = TabularDataset(
	               path=file_name, # the root directory where the data lies
	               format='json',
	                fields={'image_id': ('text', Field(sequential=True)),
             'label': ('description', Field(sequential=True)) })
	return tst

def main():
	data_root="/mnt/data2/betty/Pictures/webvision/google" #containing the json
	#test="/home/betty/Downloads/caption_datasets/dataset_flickr8k.json"
	#model = FastText.load_word2vec_format('/home/betty/Downloads/imagenet_22k_r152.vec')
	print("loading model")
	model = gensim.models.KeyedVectors.load_word2vec_format('/home/betty/Downloads/wiki-news-300d-1M-subword.vec')
	counter = Counter()
	print("model loaded with vocab", len(model.wv.vocab))
	return
	for query in sorted(os.listdir(data_root)): #args.data_path  
	        print("Processing metadata of {}".format(query))
	        with open(os.path.join(data_root,query)) as json_file:
	        	feat = json.loads(json_file.read())
	        	print("feat 0 ", feat[0]["description"])
	        	print("feat 1 ", feat[1])


	        	df = pd.DataFrame(feat)
	        	print ("df", df.head(), df.shape)	
	        	print ("df0", df["description"])
	        	stop = stopwords.words('english')
	        	#print("stopwords", stop)
	        	description=[]
	        	tokens=[]
	        	for index, row in df.iterrows():	        
        			description.append(gensim.utils.simple_preprocess(str(row["description"]).encode('utf-8').strip()))
        			counter.update(gensim.utils.simple_preprocess(str(row["description"]).encode('utf-8').strip()))
        			#tokens = nltk.tokenize.word_tokenize(row["description"].lower().decode('utf-8'))	        	
	        	print("List of lists. Let's confirm: ", type(description), " of ", type(description[0]), description)
	        	#print("output", model(df["description"][2]))
	        	#captions = []
	        	#for i, d in enumerate(feat):
	        	#	captions += [str(x['raw']) for x in d['description']]
	        	words = [word for word, cnt in counter.items() if cnt >= 0] #raise from 0 to x?
	        	vocab = Vocabulary()
	        	vocab.add_word('<pad>')
	        	vocab.add_word('<start>')
	        	vocab.add_word('<end>')
	        	vocab.add_word('<unk>')

    # Add words to the vocabulary.
	for i, word in enumerate(words):
		vocab.add_word(word)
	return vocab
	        #query_dataset = extract_feat(file_name=os.path.join(data_root, query))
	        #print("example is", query_dataset[0])

if __name__ == "__main__":
    main()
