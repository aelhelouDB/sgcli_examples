# Arctic Training Example

This example demonstrates how to run [ArcticTraining](https://github.com/snowflakedb/ArcticTraining) on Serverless GPU Compute using SGCLI.

ArcticTraining is a framework from Snowflake designed to simplify and accelerate the post-training process for large language models (LLMs). It provides modular trainer designs, simplified code structures, and integrated pipelines for creating and cleaning synthetic data.

## Prerequisites

1. A Hugging Face account with access to Llama models
2. Your HF token stored in the secrets vault at the path specified in `workload.yaml`

## Files

- `workload.yaml` - SGCLI workload configuration
- `requirements.yaml` - Python dependencies
- `recipe.yaml` - Arctic Training recipe configuration

## Usage

Run the example with:

```bash
sgcli run -f workload.yaml
```

Or watch the logs in real-time:

```bash
sgcli run -f workload.yaml --watch
```

## Customization

### Recipe Configuration

Modify `recipe.yaml` to customize your training run:

- **Model**: Change `model.name_or_path` to use a different base model
- **Data**: Modify `data.sources` to use different datasets
- **Batch size**: Adjust `micro_batch_size` and `gradient_accumulation_steps`
- **Checkpointing**: Configure `checkpoint` settings for saving

### Using Custom Datasets

You can use HuggingFace datasets with custom column mappings:

```yaml
data:
  sources:
    - type: huggingface_instruct
      name_or_path: your-org/your-dataset
      split: train
      role_mapping:
        user: instruction
        assistant: response
```

### Multi-Node Training

To scale to multiple nodes, update `workload.yaml`:

```yaml
compute:
  gpus: 16  # Total GPUs across all nodes
  gpu_type: h100

command: |-
  arctic_training /path/to/recipe.yaml \
    --num_nodes=2 \
    --num_gpus=8 \
    --no_ssh \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT
```

## Resources

- [ArcticTraining GitHub](https://github.com/snowflakedb/ArcticTraining)
- [ArcticTraining Documentation](https://arctictraining.readthedocs.io/)
- [SGCLI User Guide](https://docs.google.com/document/d/1gjwD4YiR1x8L1vZ5VzDomcUeNuMG1wVVrKsmT-M_nUU)
