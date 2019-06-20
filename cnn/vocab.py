# Create a vocabulary wrapper
import nltk
import pickle
from collections import Counter
import json
import argparse
import os


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



def from_json(path):
    dataset = json.load(open(path, 'r'))
    captions = []
    for i, d in enumerate(dataset):
        captions += [str(x['raw']) for x in d['descriptions']]
        captions += [str(x['raw']) for x in d['title']]
    return captions



def build_vocab(data_path, threshold):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    for json_file  in sorted(os.listdir(data_path)): #args.data_path  
            print("Processing metadata of {}".format(json_file))
        full_path = os.path.join(data_path, json_file)

        captions = from_json(full_path)
        
        for i, caption in enumerate(captions):
            tokens = nltk.tokenize.word_tokenize(
                caption.lower().decode('utf-8'))
            counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(data_path, data_name):
    vocab = build_vocab(data_path, threshold=4)
    with open('./vocab/%s_vocab.pkl' % data_name, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print("Saved vocabulary file to ", './vocab/%s_vocab.pkl' % data_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/w/31/faghri/vsepp_data/')
    parser.add_argument('--data_name', default='webvision')
    opt = parser.parse_args()
main(opt.data_path, opt.data_name)