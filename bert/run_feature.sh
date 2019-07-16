#export BERT_BASE_DIR=/data/tanggp/bert/multi_cased_L-12_H-768_A-12
export BERT_BASE_DIR=/data/tanggp/bert/chinese_L-12_H-768_A-12
export INPUT=/data/tanggp/xun_class/aichallenge
export OUTPUT=/data/tanggp/xun_class/bert_embedding/
#123
cp /data/tanggp/bert/chinese_L-12_H-768_A-12/bert_config.json /data/tanggp/xun_class/bert_model_mul/
cp /data/tanggp/bert/chinese_L-12_H-768_A-12/vocab.txt /data/tanggp/xun_class/bert_model_mul/
python3 extract_features2.py.py \
  --task_name=category \
  --input_file=$OINPUT/apptype_train.train_embedding\
  --output_file=$OUTPUT/apptype_train.train_jieba_json1\
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=200