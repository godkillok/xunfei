export BERT_BASE_DIR=/data/tanggp/bert_model/multi_cased_L-12_H-768_A-12
export XNLI_DIR=/data/tanggp/xfyun/classify/aichallenge
export OUTPUT=/data/tanggp/xfyun/classify/bert_model/
#123
python3 run_classify.py \
  --task_name=category \
  --do_train=true \
  --do_eval=true \
  --do_eval_pred=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --num_lables=126 \
  --learning_rate=5e-4 \
  --num_train_epochs=4.0 \
  --eval_steps=0\
  --output_dir=$OUTPUT