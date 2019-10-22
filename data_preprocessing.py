from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import nltk
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')

def bigram_train_data_save(df, output_file="./train.txt") :
    string = ""

    for i in range(0, len(df)) :
        string += " ".join(word_tokenize(df['Title'][i])) + "\n" + " ".join(word_tokenize(df['Content'][i])) + "\n"

    with open(output_file, "w") as f :
        f.write(string)
    
    del string
    
# it will used to make huffman tree 
def class_count_save(df, output_file="./data/class_vocab.json") :
    
    class_count = []
    
    for i in range(0, len(df)) :
        cls = int(df['Class'][i])
        
        while cls >= len(class_count) :
            class_count.append({
                'cn' : 0,
                'class' : len(class_count)
            })
            
        class_count[cls]['cn'] += 1
    
    # what we need is just rank
    sorted_cls = [k['cn'] for k in class_count]
    sorted_cls = sorted(sorted_cls)
    
    
    with open(output_file, 'w', encoding='utf-8') as make_file:
        json.dump([sorted_cls.index(k['cn']) for k in class_count if k['cn'] > 0], make_file, indent="\t")
    
    return sorted_cls
    
    
def make_document(df) :
    data = []
    stopWords = set(stopwords.words('english'))
    
    # bag of words 
    for i in range(0, len(df)) : 
        document = word_tokenize(df['Title'][i]) + word_tokenize(df['Content'][i])
        
        for i in range(len(document)-1, -1, -1) :
            if document[i] in stopWords :
                del document[i]                         
        
        data.append({
            "class" : int(df['Class'][i]),
            "document" : document
        })
        
    return data


def GetWordHash(word, vocab_hash) :
    hash = 1
    for a in range(0, len(word)) :
        hash = hash * 257 + ord(word[a])
    hash = hash % len(vocab_hash)
    
    return hash


def SearchVocab(word, vocab, vocab_hash) :
    hash = GetWordHash(word, vocab_hash)
    
    while True : 
        if vocab_hash[hash] == - 1 :
            return -1
        if word == vocab[vocab_hash[hash]]['word'] :
            return vocab_hash[hash]
        
        hash = (hash + 1) % len(vocab_hash)
    
    return -1


def add_bigram_features(data, vocab, vocab_hash) :
    
    for i in range(0, len(data)) :
        
        data[i]['bigram_features'] = []
        
        last_word = ""
        
        for j in range(0,len(data[i]['document'])) :
            word = data[i]['document'][j]
            bigram_word = last_word + "_" + word
            
            idx = SearchVocab(bigram_word, vocab, vocab_hash)
            if idx != -1 :
                data[i]['bigram_features'].append(bigram_word)
            
            last_word = word
            
            if SearchVocab(word, vocab, vocab_hash) == -1 :
                data[i]['document'][j] = "<unk>"
                
    return data

def main() :
    df = pd.read_csv("./data/train.csv", header=None)
    df.columns = ["Class", "Title", "Content"]
    
    data = make_document(df)
    
    del df
    
    vocab_file="./vocab/vocab.json"
    vocab_hash_file="./vocab/vocab_hash.json"
    
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    with open(vocab_hash_file, 'r') as f:
        vocab_hash = json.load(f)
    
    data = add_bigram_features(data, vocab, vocab_hash)
    
    with open('./data/train_data.json', 'w', encoding='utf-8') as make_file:
        json.dump(data, make_file, indent="\t")