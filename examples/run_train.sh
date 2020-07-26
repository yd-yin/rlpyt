#!/bin/bash

lr="2.5e-4"
gamma="0.99"
batch_size="512"
target_update_interval="1024"
eps_init="0.8"
num_train_env="9"
num_eval_env="1"

gpu_c="1"
gpu_g="0"
model_ids="Avonia,Avonia,Avonia,candcenter,candcenter,candcenter,gates_jan20,gates_jan20,gates_jan20"
model_ids_eval="Avonia"
arena="push_door"
seed="0"
model_path=""

### change default arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu_c) gpu_c="$2"; shift ;;
        --gpu_g) gpu_g="$2"; shift ;;
        --model_ids) model_ids="$2"; shift ;;
        --model_ids_eval) model_ids_eval="$2"; shift ;;
        --arena) arena="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --num_parallel) num_train_env="$2"; shift ;;
        --model_path) model_path="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

log_dir="/result"
run_ID="relmogen_dqn_"$arena"_"$seed
mkdir -p $log_dir/$run_ID

echo "run_ID:" $run_ID
echo "gpu_c:" $gpu_c
echo "gpu_g:" $gpu_g
echo "model_ids:" $model_ids
echo "model_ids_eval:" $model_ids_eval
echo "arena:" $arena
echo "seed:" $seed
echo "model_path:" $model_path

python -u example_relmogen_gibson.py \
  --gpu_c $gpu_c \
  --gpu_g $gpu_g \
  --arena $arena \
  --log_dir $log_dir \
  --run_ID $run_ID \
  --batch_size $batch_size \
  --lr $lr \
  --target_update_interval $target_update_interval \
  --model_path $model_path \
  --eps_init $eps_init \
  --num_train_env $num_train_env \
  --num_eval_env $num_eval_env \
  --model_ids $model_ids \
  --model_ids_eval $model_ids_eval > $log_dir/$run_ID/log 2>&1
