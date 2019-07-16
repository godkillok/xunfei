#export BERT_BASE_DIR=/data/tanggp/bert/multi_cased_L-12_H-768_A-12
export BERT_BASE_DIR=/data/tanggp/bert/chinese_L-12_H-768_A-12
export XNLI_DIR=/data/tanggp/xun_class/aichallenge
export OUTPUT=/data/tanggp/xun_class/bert_model/
export Tfrecord=/data/tanggp/xun_class/bert_multi_tfrecord/
export LABEL=/data/tanggp/xun_class/aichallenge/textcnn_label_sort
export LABEL_F=/data/tanggp/xun_class/aichallenge/textcnn_author_sort

python3 data_prepare.py \
  --task_name=category \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=200 \
  --train_batch_size=8 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT\
  --num_lables=2\
  --tfrecord=$Tfrecord\
    --label_path=$LABEL