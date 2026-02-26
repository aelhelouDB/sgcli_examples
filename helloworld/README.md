# Hello World Example

A minimal PyTorch training example to verify your Serverless GPU Compute setup with SGCLI.

This example trains a simple linear regression model to learn `y = 2x + 1` and logs metrics to MLflow.

## Files

- `workload.yaml` - SGCLI workload configuration
- `requirements.yaml` - Python dependencies (mlflow)
- `train.py` - Simple PyTorch training script

## Usage

Run the example:

```bash
sgcli run -f workload.yaml
```

Or watch the logs in real-time:

```bash
sgcli run -f workload.yaml --watch
```

## What It Does

1. Creates synthetic data following `y = 2x + 1 + noise`
2. Trains a single-layer linear model using SGD
3. Logs loss metrics to MLflow every 20 epochs
4. Prints learned weight (~2.0) and bias (~1.0) to verify training worked

## Expected Output

```
Using device: cuda
Epoch [20/100], Loss: 0.0xyz
Epoch [40/100], Loss: 0.0xyz
...
Learned weight: ~2.0000 (expected: ~2.0)
Learned bias: ~1.0000 (expected: ~1.0)
Training complete!
```

## Customization

### Single-Node Training

For single-node with 1 GPU:

```yaml
compute:
  gpus: 1
  gpu_type: a10

command: |-
  python /path/to/train.py
```

### Multi-Node Training (Current)

The current config uses 2 A10 GPUs across 2 nodes:

```yaml
compute:
  gpus: 2
  gpu_type: a10

command: |-
  torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    ...
```

## Resources

- [SGCLI User Guide](https://docs.google.com/document/d/1gjwD4YiR1x8L1vZ5VzDomcUeNuMG1wVVrKsmT-M_nUU)
