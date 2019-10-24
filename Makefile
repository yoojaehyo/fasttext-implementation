run_fasttext:
		CUDA_VISIBLE_DEVICES=1 python fasttext.py \
		--vocab_file=./vocab/vocab.json \
		--vocab_hash_file=./vocab/vocab_hash.json \
		--do_train=True \
		--train_file=./data/train_data.json \
		--do_predict=True \
		--predict_file=./data/test_data.json \
		--train_batch_size=32 \
		--predict_batch_size=32 \
		--num_class=4 \
		--learning_rate=0.25 \
		--num_epochs=5.0 \
		--hidden_size=10 \
		--embedding_size=300 \
		--output_dir=./out/fasttext \

data_loader_test:
		CUDA_VISIBLE_DEVICES=1 python data_loader_test.py \
		--vocab_file=./vocab/vocab.json \
		--vocab_hash_file=./vocab/vocab_hash.json \
		--train_file=./data/train_data.json \
		--predict_file=./data/test_data.json \