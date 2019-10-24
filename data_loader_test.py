from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import tensorflow as tf

import fasttext

flags = tf.flags

FLAGS = flags.FLAGS


def main(_) :

	with open(FLAGS.vocab_file, 'r') as f:
		vocab = json.load(f)
	with open(FLAGS.vocab_hash_file, 'r') as f:
		vocab_hash = json.load(f)

	with open(FLAGS.predict_file, 'r') as f:
		data = json.load(f)
		data = data[0:100]
		num_examples = len(data)

	batch_size = 32

	raw_data, raw_len, raw_labels = fasttext.raw_data_load(data, vocab, vocab_hash)

	loader = fasttext.input_fn(raw_data, raw_len, raw_labels, batch_size, len(vocab))

	print("\n\n\n")

	i=0
	for data, labels in loader :
		print(data["input_tensor"])
		print(labels)
		i+=1
		if i == 100:
			break

if __name__ == '__main__':
	tf.enable_eager_execution()
	tf.app.run()