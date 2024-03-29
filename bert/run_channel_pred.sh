export BERT_BASE_DIR=/data/tanggp/bert_model/multi_cased_L-12_H-768_A-12
export XNLI_DIR=/data/tanggp/nsfw_txt_input
export OUTPUT=/data/tanggp/nsfw_txt_output/
#12
python3 bert_pred.py \
  --task_name=nsfw \
  --do_train=true \
  --do_eval=true \
  --do_eval_pred=true \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=3.0 \
  --eval_steps=0\
  --output_dir=$OUTPUT