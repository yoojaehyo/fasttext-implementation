from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
	"train_file", None,
	"json file for training.")

flags.DEFINE_string(
    "predict_file", None,
    "json file for prediction.")

flags.DEFINE_integer(
    "embedding_size", 300,
    "embedgging size for word")

flags.DEFINE_string(
	"NUM_CLASS", 4,
	"json file for training.")

flags.DEFINE_integer(
    "hidden_size", 10,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")



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
        if word == vocab[vocab_hash[hash]].word :
            return vocab_hash[hash]
        
        hash = (hash + 1) % vocab_hash_size
    
    return -1




def modeling(input_tensor, labels, num_class, vocab_size, embedding_size, hidden_size, n_class) :
	'''
		returns result of Fasttext modeling

		Args : 
			input_tensor : BOW of data tensor of shape [N (num_document), vocab_size]
			labels : label class tensor of shape [N]
			num_class : number of class to predict
			vocab_size : length of n-gram vocab
			embedding_size : size of word embedding
			n_class : number of output classes

	'''

	with tf.variable_scope("word_embedding") :
		embedding_table = tf.get_variable(
			name="embedding_table",
			shape=[vocab_size, embedding_size])

	# # add all the words and divide by its count
	# 'Ax' = [N, embedding_size]
	Ax = tf.matmul(input_tensor, embedding_table)
	word_count = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
	Ax = Ax / word_count

	with tf.variable_scope("hidden_layer") :
		# 'BAX' = [N, hidden_size]
		BAx = tf.layers.dense(
				Ax,
				hidden_size,
				activation=None,
				name="hidden_weight")

	with tf.variable_scope("softmax_loss") :

		logit = tf.layers.dense(
					input_tensor,
					n_class,
					activation=None,
					name="softmax_weight")

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

	return loss







