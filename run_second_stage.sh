CUDA_VISIBLE_DEVICES=3 nohup  python /root/EarlyRobust/second_stage.py\
       --model_name_or_path /root/EarlyRobust/imdb/first_stage_models/checkpoint-0\
      --dataset_name imdb       --task_name None       --num_examples 500\
     --logging_steps 1000       --do_train      --save_steps 20000\
      --max_seq_length 256       --per_device_train_batch_size 8       --learning_rate 2e-5\
       --num_train_epochs 10       --seed 42       --overwrite_output_dir   \
         --self_pruning_ratio   0.166667         --self_pruning_method  global\
         --inter_pruning_ratio  0.4        --inter_pruning_method global     \
           --slimming_coef_step   625         \
            --self_slimming_coef_file  /root/EarlyRobust/imdb/first_stage_models/self_slimming_coef_records.npy \
              --inter_slimming_coef_file /root/EarlyRobust/imdb/first_stage_models/inter_slimming_coef_records.npy \
               --prune_before_train true       --data_dir .cache/huggingface/datasets/glue/sst2    \
               --num_labels 2       --acc_threshold 0.9          \
             --output_dir /root/EarlyRobust/imdb/second_stage_models \
              > /root/EarlyRobust/imdb/second_stage_log.log 2>&1  &
