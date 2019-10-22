# Reference From https://github.com/wlin12/wang2vec/blob/master/word2phrase.c

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

vocab_hash_size = 100000000 #100M for bigram
vocab_hash = [int(-1)] * vocab_hash_size

train_file = None

vocab = []
vocab_size = 0

min_count = 5
min_reduce = 1
unk_num = 0

train_words = 0


# Returns hash value of a word
def GetWordHash(word) :
    
    hash = 1
    for a in range(0, len(word)) :
        hash = hash * 257 + ord(word[a])
    hash = hash % vocab_hash_size
    return hash


# Returns position of a word in the vocabulary; if the word is not found, returns -1
def SearchVocab(word) :
    hash = GetWordHash(word)
    while True : 
        if vocab_hash[hash] == -1 :
            return -1
        if word == vocab[vocab_hash[hash]]['word'] :
            return vocab_hash[hash]
        hash = (hash + 1) % vocab_hash_size
    
    return -1


# Adds a word to the vocabulary
def AddWordToVocab(word) :
    global vocab_size, vocab, vocab_hash_size, vocab_hash
    
    # 이 부분이 C 에비해 심각하게 시간이 오래걸릴것으로 예상...
    vocab.append({"cn" : 0, "word" : word})
    vocab_size += 1
    
    hash = GetWordHash(word)
    
    while vocab_hash[hash] != -1 :
        hash = (hash + 1) % vocab_hash_size
    
    vocab_hash[hash] = vocab_size - 1
    
    # return Word index
    return vocab_size - 1
        

# Sorts the vocabulary by frequency using word counts
def SortVocab() :
    global vocab_size, vocab, vocab_hash_size, vocab_hash, min_count, unk_num
    
    # Sort the vocabulary and keep </s> at the first position
    vocab = sorted(vocab, key=lambda k : k['cn'], reverse=True)
    
    for a in range(0, vocab_hash_size) :
        vocab_hash[a] = -1
        
        
    b = vocab_size
    for a in range(0, vocab_size) : 
        # Words occuring less than min_count times will be discarded from the vocab
        if vocab[a]["cn"] < min_count :
            b = a
            break
        else :
            # Hash will be re-computed as after the sorting it is not actual
            hash = GetWordHash(vocab[a]['word'])
            while vocab_hash[hash] != -1 :
                hash = (hash + 1) % vocab_hash_size
            vocab_hash[hash] = a
                
    for a in range(b, vocab_size) :
        # deletion like tetris
        del vocab[b]
        unk_num += 1
    
    vocab_size = b


# Reduces the vocabulary by removing infrequent tokens
def ReduceVocab() :
    global vocab_size, vocab, vocab_hash_size, vocab_hash, min_reduce, unk_num
    
    b = 0
    for a in range(0, vocab_size) :
        if vocab[a]["cn"] > min_reduce :
            vocab[b]["cn"] = vocab[a]["cn"]
            vocab[b]['word'] = vocab[a]['word']
            b += 1
    
    # delete reduced vocab
    for a in range(b, vocab_size) :
        # deletion like tetris
        del vocab[b]
        unk_num += 1
    
    vocab_size = b
    for a in range(0, vocab_hash_size) :
        vocab_hash[a] = -1
    
    for a in range(0, vocab_size) :
        # Hash will be re-computed as it is not actual
        hash = GetWordHash(vocab[a]['word'])
        while vocab_hash[hash] != -1 :
            hash = (hash + 1) % vocab_hash_size
        vocab_hash[hash] = a
    
    min_reduce += 1
    

def LearnVocabFromTrainFile() :
    global vocab_size, vocab, vocab_hash_size, vocab_hash, train_words, train_file
    
    word = ""
    last_word = ""
    bigram_word = ""
    
    fin = open(train_file, "r")
    
    vocab_size = 0
    
    for line in fin.readlines() :
        words = line.split()
        
        start = 1
        
        for word in words : 
            word = word.strip()

            train_words += 1

            if train_words % 100000 == 0 :
                print("Words processed : %dK        Vocab Size : %dK" % (train_words/1000, vocab_size / 1000), flush=True)

            i = SearchVocab(word)
            if i == -1 :
                a = AddWordToVocab(word)
                vocab[a]["cn"] = 1
            else :
                vocab[i]["cn"] += 1

            if start == 1 :
                last_word = word
                start = 0
                continue

            bigram_word = last_word + "_" + word
            last_word = word

            i = SearchVocab(bigram_word)
            if i == -1 :
                a = AddWordToVocab(bigram_word)
                vocab[a]["cn"] = 1
            else :
                vocab[i]["cn"] += 1

            # for good hashing
            if vocab_size > vocab_hash_size * 0.7 :
                ReduceVocab()
            
    SortVocab()
    a = AddWordToVocab("<unk>")
    vocab[a]['cn'] = unk_num
    print("\nVocab size (unigrams + bigrams): %d\n" % vocab_size, flush=True)
    print("Words in train file: %d\n" % train_words, flush=True)
    
    fin.close()
    

def TrainModel() :
    global train_file
    
    print("Starting training using file %s\n" % train_file)
    LearnVocabFromTrainFile()
    
    global vocab, vocab_hash
    
    with open('./vocab/vocab.json', 'w', encoding='utf-8') as make_file:
        json.dump(vocab, make_file, indent="\t")
        
    with open('./vocab/vocab_hash.json', 'w', encoding='utf-8') as make_file:
        json.dump(vocab_hash, make_file, indent="\t")

def main() :
    global train_file
    
    train_file = "./train.txt"
    
    TrainModel()