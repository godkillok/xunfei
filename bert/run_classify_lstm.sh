#export BERT_BASE_DIR=/data/tanggp/bert/multi_cased_L-12_H-768_A-12
export BERT_BASE_DIR=/data/tanggp/bert/chinese_L-12_H-768_A-12
export XNLI_DIR=/data/tanggp/xun_class/bert_multi_tfrecord/
export OUTPUT=/data/tanggp/xun_class/bert_model_mul/
export HISTORY=/data/tanggp/xun_class/aichallenge/bert_multi_history
export LABEL=/data/tanggp/xun_class/aichallenge/textcnn_label_sort
export LABEL_F=/data/tanggp/xun_class/aichallenge/textcnn_author_sort
#123
python3 run_classify_lstm.py \
  --task_name=category \
  --do_train=true \
  --do_eval=false \
  --do_predict=false\
  --do_eval_pred=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=200 \
  --train_batch_size=16 \
  --num_lables=125 \
  --learning_rate=1e-5 \
  --num_train_epochs=6.0 \
  --eval_steps=0\
  --output_dir=$OUTPUT\
  --history_dir=$HISTORY\
  --label_path=$LABEL\
  --labels_first_path=$LABEL_F