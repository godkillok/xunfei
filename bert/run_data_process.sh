export BERT_BASE_DIR=/data/tanggp/bert_model/multi_cased_L-12_H-768_A-12
export XNLI_DIR=/data/tanggp/xun_class/aichallenge
export OUTPUT=/data/tanggp/xun_class/bert_model/
python3 data_prepare.py \
  --task_name=category \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT\
  --num_lables=2
