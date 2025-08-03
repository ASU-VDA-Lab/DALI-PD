# src

This directory contains scripts for running **DALI-PD**.

## Ablation Study Experiments

[`ablation_ir.py`](./ablation_ir.py) and [`ablation_rudy.py`](./ablation_rudy.py) show how to train the model using either CircuitNet’s real circuit heatmaps or DALI-PD’s synthetic heatmaps to predict:  
(1) **RUDY** and (2) **IR drop**.

The arguments are listed below:

- `--device`: The CUDA device ID.
- `--load_weight_path`: Path to load the model weights. Use this if you first pretrain the model with DALI-PD’s synthetic heatmaps and then finetune it on CircuitNet’s real circuit heatmaps.
- `--save_weight_path`: Path to save the model weights.
- `--CircuitNet_test_path`: Path to CircuitNet’s test set based on our setup. Please refer to Table 1.
- `--CircuitNet_train_path`: Path to CircuitNet’s training set based on our setup. Please refer to Table 1.
- `--synthetic_path`: Path to the DALI-PD dataset in [`synthetic_benchmark`](../synthetic_benchmark).
- `--learning_rate`: Learning rate (default: `5e-5`).
- `--weight_decay`: Weight decay rate (default: `5e-6`).
- `--gradient_accum_steps`: Number of gradient accumulation steps to simulate batch training. Adjust based on the dataset.
- `--steps`: Number of training steps. Set to `125` for pretraining, and `25` for finetuning.
- `--train_with_synthetic`: Set to `True` to train with DALI-PD’s synthetic dataset, or `False` to train with CircuitNet’s real circuit heatmaps.
- `--num_of_heatmap`: Number of heatmaps to sample.
