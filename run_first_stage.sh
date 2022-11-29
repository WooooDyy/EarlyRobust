CUDA_VISIBLE_DEVICES=7 nohup  python /root/EarlyRobust/freelb_search.py \
  --model_name bert-base-uncased       --dataset_name imdb      --task_name None \
  --save_steps 2500       --max_seq_length 256       --bsz 8       --lr 2e-5 \
  --seed 42       --l1_loss_self_coef  1e-5       --l1_loss_inter_coef 2e-4 \
  --max_epochs 1      --adv_steps 5       --adv_lr 0.01      --adv_init_mag 0.05 \
  --adv_max_norm 0      --adv_norm_type l2       --adv_change_rate 0.05     --max_grad_norm 1 \
  --epochs 1        --num_labels   2 \
  --output_dir /root/EarlyRobust/imdb/first_stage_models > /root/EarlyRobust/imdb/first_stage.log 2>&1 &