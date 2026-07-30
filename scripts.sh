export CUDA_VISIBLE_DEVICES=0

# FSE is a post-hoc, two-stage plugin:
#   Stage 1: train the baseline forecaster normally and freeze it.
#   Stage 2: train FSE (teacher/student dual-alignment + guided spectral
#            correction) on top of the frozen baseline checkpoint.
#
# The two stages MUST share seq_len/label_len/pred_len/e_layers/d_layers/
# d_model/n_heads/d_ff/expand/d_conv/factor/embed/distil/enc_in/dec_in/c_out,
# since FSE reconstructs the baseline internally from these configs and loads
# its state_dict directly.

model_name=DLinear
data_id=ETTh1_96_96

# ---- Stage 1: pretrain & freeze the baseline forecaster ----
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id $data_id \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1

# Reproduces run.py's setting-name convention for the run above (itr index 0),
# so we can point stage 2 at the exact checkpoint that was just saved.
baseline_setting="long_term_forecast_${data_id}_${model_name}_ETTh1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"
baseline_ckpt="./checkpoints/${baseline_setting}/checkpoint.pth"

# ---- Stage 2: train FSE on top of the frozen baseline ----
python -u run.py \
  --task_name fse_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ${data_id}_FSE \
  --model FSE \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --fse_baseline $model_name \
  --fse_baseline_ckpt $baseline_ckpt \
  --fse_layers 3 \
  --fse_alpha 0.3 \
  --fse_eps 1e-6 \
  --fse_gamma 1.0 \
  --fse_eta 0.8 \
  --fse_lambda_cos 0.8 \
  --fse_lambda_ema 0.2 \
  --fse_ema_beta 0.8 \
  --learning_rate 0.0001 \
  --lradj cosine \
  --patience 3 \
  --des 'Exp' \
  --itr 1
