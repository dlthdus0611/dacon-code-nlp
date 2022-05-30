python train.py \
    --model_name_or_path microsoft/graphcodebert-base \
    --train_file ../data/code2df_simcse.csv \
    --output_dir result/my-unsup-simcse-codebert-base \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 256 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"

python train.py \
    --model_name_or_path microsoft/graphcodebert-base \
    --train_file ../data/code2df_simcse.csv \
    --output_dir result/my-unsup-codebert-base_epoch3 \
    --dataloader_drop_last \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 256 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"