from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import numpy as np
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string("vocab_file", None,
					"The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("vocab_hash_file", None,
					"The vocabulary hash file that speeds up to search vocab")

flags.DEFINE_string(
	"output_dir", None,
	"The output directory where the model checkpoints will be written.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
	"How often to save the model checkpoint.")

flags.DEFINE_bool(
	"do_train", True,
	"True if do train")

flags.DEFINE_string(
	"train_file", None,
	"json file for training.")

flags.DEFINE_float(
	"learning_rate", 1e-2,
	"learning_rate for training")

flags.DEFINE_float(
	"decay_rate", 5e-5,
	"True if do train")

flags.DEFINE_bool(
	"do_predict", True,
	"True if do predict")

flags.DEFINE_string(
	"predict_file", None,
	"json file for prediction.")

flags.DEFINE_integer(
	"embedding_size", 300,
	"embedgging size for word")

flags.DEFINE_integer(
	"num_class", 4,
	"json file for training.")

flags.DEFINE_integer(
	"hidden_size", 10,
	"When splitting up a long document into chunks, how much stride to "
	"take between chunks.")

flags.DEFINE_float(
	"num_epochs", 5.0,
	"number of epochs")

flags.DEFINE_integer(
	"train_batch_size", 32,
	"number of batch_size for train")

flags.DEFINE_integer(
	"predict_batch_size", 32,
	"number of batch_size for predict")


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
			return SearchVocab("<unk>", vocab, vocab_hash)
		if word == vocab[vocab_hash[hash]]['word'] :
			return vocab_hash[hash]
		
		hash = (hash + 1) % len(vocab_hash)
	
	return SearchVocab("<unk>", vocab, vocab_hash)



def modeling(input_tensor, labels, vocab_size, embedding_size, hidden_size, n_class, mode) :
	'''
		returns result of Fasttext modeling

		Args : 
			input_tensor : dow of each document of shape [N, vocab_size],
			labels : label class tensor of shape [N, 1]
			vocab_size : size of n-gram vocab
			embedding_size : size of word embedding
			n_class : number of output classes

	'''

	with tf.variable_scope("word_embedding") :
		embedding_table = tf.get_variable(
			name="embedding_table",
			shape=[vocab_size, embedding_size],
			dtype=tf.float32)

	# 'Ax' = [N, embedding_size]
	Ax = tf.matmul(input_tensor, embedding_table)

	with tf.variable_scope("hidden_layer") :
		# 'BAX' = [N, hidden_size]
		BAx = tf.layers.dense(
				Ax,
				hidden_size,
				activation=None,
				name="hidden_weight")

	with tf.variable_scope("softmax_loss") :
		# 'logits' = [N, n_class]
		logits = tf.layers.dense(
					BAx,
					n_class,
					activation=None,
					name="softmax_weight")

		# only logits needed for predict
		if mode == tf.estimator.ModeKeys.PREDICT :
			return logits, -1

		# 'labels' => [N, n_class]
		labels = tf.reshape(labels, [-1])
		labels = tf.one_hot(labels, n_class)

		logSF = tf.nn.log_softmax(logits)
		loss = -tf.reduce_mean(tf.reduce_sum(labels*logSF, axis=-1))

	return logits, loss


def model_fn(features, labels, mode, params) : 

	logit, loss = modeling(input_tensor=features["input_tensor"],
				labels=labels,
				vocab_size=params['vocab_size'],
				embedding_size=params['embedding_size'],
				hidden_size=params['hidden_size'],
				n_class=params['n_class'],
				mode=mode)

	output_spec = None

	if mode == tf.estimator.ModeKeys.TRAIN :
		global_step=tf.train.get_global_step()

		# linearly decaying lr
		learning_rate = (1 - tf.cast(global_step, tf.float32) * tf.constant(FLAGS.decay_rate)) * tf.constant(FLAGS.learning_rate)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(loss, global_step=global_step)

		output_spec = tf.estimator.EstimatorSpec(
					mode=mode,
					loss=loss,
					train_op=train_op)
		
	elif mode == tf.estimator.ModeKeys.PREDICT :
		predicted_labels = tf.argmax(logit, axis=-1)

		predictions = {
			"labels" : tf.reshape(features["labels"], [-1]),
			"labels_p" : predicted_labels
		}

		output_spec = tf.estimator.EstimatorSpec(
					mode=mode,
					predictions=predictions)

	else :
		raise ValueError("Only Train or Predict modes are supported: %s" % (mode))

	return output_spec


def input_fn(raw_inputs, raw_len, raw_labels, batch_size, vocab_size, is_training, drop_remainder) :

	inputs = {	
		"inputs" : raw_inputs,
		"length" : raw_len,
		"labels" : raw_labels # for predictions...
	}

	def make_bow_py(idx_list, length) :
		bow = np.zeros((idx_list.shape[0], vocab_size), dtype=np.float32)

		for i in range(0, len(idx_list)) :
			for j, idx in enumerate(idx_list[i]) :
				if j == length[i] :
					break
				bow[i][idx] += 1.0 / np.float(length[i])

		return bow

	def make_bow(features, labels):
		features["input_tensor"] = tf.py_func(make_bow_py, [features['inputs'], features['length']], tf.float32)

		return features, labels

	dataset = tf.data.Dataset.from_tensor_slices((inputs, raw_labels))

	if is_training :
		dataset=dataset.repeat().shuffle(40000)
	
	dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
	dataset = dataset.map(make_bow)

	return dataset


def raw_data_load(data, vocab, vocab_hash) :

	raw_data = [[SearchVocab(word, vocab, vocab_hash)
			for word in data[i]['document'] + data[i]['bigram_features']]
				for i in range(0, len(data))]

	raw_len = [len(w) for w in raw_data]
	max_len = max([len(w) for w in raw_data])

	# padding with 0
	for i in range(0, len(raw_data)) :
		raw_data[i] = raw_data[i] + ([0] * (max_len - raw_len[i]))

	# since true label starts at 1...
	raw_labels = [[data[i]['class']-1] for i in range(0, len(data))]

	return raw_data, raw_len, raw_labels


def main(_) :
	tf.logging.set_verbosity(tf.logging.INFO)

	with open(FLAGS.vocab_file, 'r') as f:
		vocab = json.load(f)
	with open(FLAGS.vocab_hash_file, 'r') as f:
		vocab_hash = json.load(f)

	vocab_size = len(vocab)

	tf.gfile.MakeDirs(FLAGS.output_dir)
 
	run_config = tf.estimator.RunConfig(
			model_dir=FLAGS.output_dir,
			save_checkpoints_steps=FLAGS.save_checkpoints_steps)

	estimator = tf.estimator.Estimator(
			model_fn=model_fn,
			config=run_config,
			params={
				"vocab_size" : vocab_size,
				"embedding_size" : FLAGS.embedding_size,
				"hidden_size" : FLAGS.hidden_size,
				"n_class" : FLAGS.num_class
			})


	if FLAGS.do_train :
		with open(FLAGS.train_file, 'r') as f:
			data = json.load(f)
			data = data
			num_examples = len(data)

		batch_size = FLAGS.train_batch_size

		tf.logging.info("Loading Train data...")
		raw_data, raw_len, raw_labels = raw_data_load(data, vocab, vocab_hash)
		train_steps = int(float(num_examples) / float(batch_size) * FLAGS.num_epochs)

		tf.logging.info("\n\n***** Running training *****")
		tf.logging.info("  Num examples = %d", num_examples)
		tf.logging.info("  Batch size = %d", batch_size)
		tf.logging.info("  Num steps = %d", train_steps)

		current_time = time.time()
		estimator.train(
				input_fn=lambda: input_fn(raw_data, raw_len, raw_labels, batch_size, vocab_size, is_training=True, drop_remainder=True),
				max_steps=train_steps)

		tf.logging.info("Trainning time : %.2f minutes\n\n\n", ((time.time() - current_time) / 60.0))
		

	if FLAGS.do_predict :
		with open(FLAGS.predict_file, 'r') as f:
			data = json.load(f)
			data = data
			num_examples = len(data)

		batch_size = FLAGS.predict_batch_size

		tf.logging.info("Loading Predict data...")
		raw_data, raw_len, raw_labels = raw_data_load(data, vocab, vocab_hash)

		tf.logging.info("\n\n***** Running Predictions *****")
		tf.logging.info("  Num examples = %d", num_examples)
		tf.logging.info("  Batch size = %d", batch_size)

		T = 0
		F = 0
		for predictions in estimator.predict(
			input_fn=lambda: input_fn(raw_data, raw_len, raw_labels, batch_size, vocab_size, is_training=False, drop_remainder=False),
			yield_single_examples=False):

			for i in range(0, len(predictions['labels_p'])) :
				if (T + F) % 1000 == 0 :
					tf.logging.info("Processing example: %d" % (T+F))

				if predictions['labels_p'][i] == predictions['labels'][i] :
					T+=1
				else :
					F+=1

		tf.logging.info("Accuracy : %.3f", (T/(T+F)))


if __name__ == "__main__":
	flags.mark_flag_as_required("vocab_file")
	flags.mark_flag_as_required("vocab_hash_file")
	flags.mark_flag_as_required("output_dir")
	tf.app.run()


