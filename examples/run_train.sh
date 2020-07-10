#!/bin/bash

python -u example_relmogen_gibson.py \
  --cuda_idx 1 \
  --arena push_drawers \
  --run_ID push_drawers_shallow_net_bz_1024_lr_1e-3_tgt_update_1024_eps_init_0.8_task_simplified_only_explore_free_space_debugged \
  --batch_size 512 \
  --lr 1e-3 \
  --target_update_interval 1024 \
  --eps_init 0.8
