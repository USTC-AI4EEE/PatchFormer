#export CUDA_VISIBLE_DEVICES=1

model_name=PatchFormer
for test_name in CS2_35
# CS2_35 CS2_36 CS2_37 CS2_38   
do
  python -u RUL_Prediction_PatchFormer.py \
    --model $model_name \
    --root_dir 'CALCE_RUL_prediction_sl_64' \
    --seq_len 64 \
    --pred_len 1 \
    --patch_len 2 \
    --d_model 16 \
    --count 10 \
    --batch_size 128 \
    --test_name $test_name \
    --max_epochs 200
done



