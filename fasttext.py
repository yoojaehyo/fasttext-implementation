from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
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
	"do_train", False,
	"True if do train")

flags.DEFINE_string(
	"train_file", None,
	"json file for training.")

flags.DEFINE_float(
	"learning_rate", 1e-2,
	"learning_rate for training")

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

flags.DEFINE_integer(
    "num_epoch", 5,
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



def modeling(input_tensor, labels, vocab_size, embedding_size, hidden_size, n_class) :
	'''
		returns result of Fasttext modeling

		Args : 
			input_tensor : BOW of data tensor of shape [N (num_document), vocab_size]
			labels : label class tensor of shape [N]
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

	return logit, loss


def model_fn(features, labels, mode, params) : 

	logit, loss = modeling(input_tensor=features,
				labels=labels,
				vocab_size=params['vocab_size'],
				embedding_size=params['embedding_size'],
				hidden_size=params['hidden_size'],
				n_class=params['n_class'])

	output_spec = None

	if mode == tf.estimator.ModeKeys.TRAIN : 
		optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
		train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

		output_spec = tf.estimator.EstimatorSpec(
					mode=mode,
					loss=loss,
					train_op=train_op)
		
	elif mode == tf.estimator.ModeKeys.PREDICT : 
		predicted_labels = tf.argmax(logit, axis=-1)

		predictions = {
			"labels" = labels,
			"labels_p" : predicted_labels
		}

		output_spec = tf.estimaor.EstimatorSpec(
					mode=mode,
					predictions=predictions)

	else :
		raise ValueError("Only Train or Predict modes are supported: %s" % (mode))

	return output_spec


def input_fn(raw_inputs, raw_labels, batch_size) :
	
	dataset = tf.data.Dataset.from_tensor_slices((raw_inputs, raw_labels))
	dataset.repeat().shuffle(100).batch(batch_size)

	return dataset


def raw_data_load(data, vocab, vocab_hash) :

	raw_data = [[SearchVocab(word, vocab, vocab_hash)				\
			for word in data[i]['document'] + data[i]['bigram_features']]
				for i in range(0, len(data))]

	raw_labels = [data[i]['class'] for i in range(0, len(data))]

	return raw_data, raw_labels


def main() :
    
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    with open(vocab_hash_file, 'r') as f:
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
    		num_examples = len(data)

    	batch_size = FLAGS.train_batch_size

    	tf.logging.info("Loading Train data...")
    	raw_data, raw_labels = raw_data_load(data, vocab, vocab_hash)
    	train_steps = int(num_examples / batch_size * FLAGS.num_epoch)

		tf.logging.info("\n\n***** Running training *****")
		tf.logging.info("  Num examples = %d", num_examples)
		tf.logging.info("  Batch size = %d", batch_size)
		tf.logging.info("  Num steps = %d", train_steps)

    	estimator.train(
    		input_fn=lambda: input_fn(raw_data, raw_labels, batch_size),
    		max_steps=train_steps)


    if FLAGS.do_predict :
    	with open(FLAGS.predict_file, 'r') as f:
    		data = json.load(f)
    		num_examples = len(data)

    	batch_size = FLAGS.predict_batch_size

    	tf.logging.info("Loading Predict data...")
    	raw_data, raw_labels = raw_data_load(data, vocab, vocab_hash)

		tf.logging.info("\n\n***** Running Predictions *****")
		tf.logging.info("  Num examples = %d", num_examples)
		tf.logging.info("  Batch size = %d", batch_size)

		T = 0
		F = 0
    	for predictions in estimator.predict(
    		input_fn=lambda: input_fn(raw_data, raw_labels, batch_size)):
    		if predictions['labels_p'] == predictions['labels'] :
    			T+=1
    		else :
    			F+=1

    	tf.logging.info("@Accuracy : %.3f %", (T/(T+F)))


if __name__ == "__main__":
	flags.mark_flag_as_required("vocab_file")
	flags.mark_flag_as_required("vocab_hash_file")
	flags.mark_flag_as_required("output_dir")
	tf.app.run()


