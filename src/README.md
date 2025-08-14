# src

This directory contains scripts for running **DALI-PD**.

## Ablation Study Experiments

[`ablation_ir.py`](./ablation_ir.py) and [`ablation_rudy.py`](./ablation_rudy.py) show how to train the model using either [`CircuitNet`](https://circuitnet.github.io/)’s real circuit heatmaps or DALI-PD’s synthetic heatmaps to predict:  
(1) **RUDY** and (2) **IR drop**.

The arguments are listed below:

- `--device`: The CUDA device ID.
- `--load_weight_path`: Path to load the model weights. Use this if you first pretrain the model with DALI-PD’s synthetic heatmaps and then finetune it on CircuitNet’s real circuit heatmaps.
- `--save_weight_path`: Path to save the model weights.
- `--CircuitNet_test_path`: Path to [`CircuitNet`](https://circuitnet.github.io/)’s test set based on our setup. Please refer to Table 1.
- `--CircuitNet_train_path`: Path to [`CircuitNet`](https://circuitnet.github.io/)’s training set based on our setup. Please refer to Table 1.
- `--synthetic_path`: Path to the DALI-PD dataset in [`synthetic_benchmark`](../synthetic_benchmark).
- `--learning_rate`: Learning rate (default: `5e-5`).
- `--weight_decay`: Weight decay rate (default: `5e-6`).
- `--gradient_accum_steps`: Number of gradient accumulation steps to simulate batch training. Adjust based on the dataset.
- `--steps`: Number of training steps. Set to `125` for pretraining, and `25` for finetuning.
- `--train_with_synthetic`: Set to `True` to train with DALI-PD’s synthetic dataset, or `False` to train with [`CircuitNet`](https://circuitnet.github.io/)’s real circuit heatmaps.
- `--num_of_heatmap`: Number of heatmaps to sample.

## Parameters for reprodcing the results in the paper:

### Pretraining (for both IR Drop and RUDY prediction tasks)
| Dataset   | Batch Size | Steps |
|-----------|------------|-------|
| CircuitNet| 64         | 125   |
| DALI-PD   | 64         | 125   |

### With limited CircuitNet data (for pretraining from scratch and fine-tuning the DALI-PD-trained model on both IR Drop and RUDY prediction tasks)

![Pretrain_finetune](etc/pretrain_finetune.png)

### Step Scaling Rule

The number of training steps scales with the **size of the CircuitNet dataset** according to the formula:

Steps ≈ 1.75 × sqrt(Dataset Size)

---

### Dataset Size vs. Training Configuration

| Data Count | Batch Size | Epochs | Steps  |
|------------|------------|--------|--------|
| 50         | 1          | ~1   | 13  |
| 100        | 2          | ~1   | 18  |
| 200        | 2          | ~1   | 25  |
| 300        | 4          | ~1   | 31  |
| 400        | 4          | ~1   | 35  |
| 500        | 8          | ~1   | 40  |
| 600        | 8          | ~1   | 43  |
| 700        | 8          | ~1   | 47  |
| 800        | 8          | ~1   | 50  |
| 900        | 16         | ~1   | 53  |
| 1000       | 16         | ~1   | 55  |
| 1250       | 16         | ~1   | 62  |
| 1500       | 16         | ~1   | 68  |
| 1750       | 32         | ~1   | 73  |
| 2000       | 32         | ~1   | 79  |
| 2861       | 64         | ~2   | 94  |