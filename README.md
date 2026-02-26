# SGCLI Examples

This directory contains example configurations for launching training workloads on Serverless GPU Compute using the command-line interface (SGCLI).

For detailed user documentation, see the [SGCLI User Guide](https://docs.google.com/document/d/1gjwD4YiR1x8L1vZ5VzDomcUeNuMG1wVVrKsmT-M_nUU/edit?usp=sharing).

For questions or support, please contact ben.hansen@databricks.com & amine.elhelou@databricks.com

**Limitations: **
- Launching ray jobs via the `sgcli` command is not supported yet but can be done interactively via notebooks or databricks jobs see examples [here](./ray_notebook_examples/)

## Examples

| Example | Description | GPUs |
|---------|-------------|------|
| [helloworld](./helloworld/) | Minimal PyTorch training example to verify setup | 2x A10 |
| [trl-sft](./trl/) | Fine-tune Gemma 3 with HuggingFace TRL SFTTrainer and LoRA | 16x H100 |
| [axolotl](./axolotl/) | Fine-tune Llama-3.1-8B with Axolotl using FSDP and Liger kernels | 16x H100 |
| [arctic_training](./arctic_training/) | Fine-tune LLMs with Snowflake's Arctic Training framework | 8x H100 |
| [lingua](./lingua/) | Train language models with Meta's Lingua library | 8x H100 |
| [verl](./verl/) | Reinforcement learning fine-tuning with veRL (GRPO/PPO) | 8x H100 |

## Structure

Each subdirectory contains:

- `workload.yaml` - SGCLI workload configuration
- `requirements.yaml` - Python dependencies
- `README.md` - Example-specific documentation

Some examples include additional config files (e.g., `recipe.yaml`, `train.py`).

## Quick Start

1. Navigate to an example directory:
   ```bash
   cd helloworld
   ```

2. Run the workload:
   ```bash
   sgcli run -f workload.yaml
   ```

3. Watch logs in real-time:
   ```bash
   sgcli run -f workload.yaml --watch
   ```

## Common Configuration

### Environment Variables

Most examples use these environment variables:

```yaml
environment:
  env_variables:
    NCCL_DEBUG: "INFO"
    MLFLOW_TRACKING_URI: "databricks"
  env_variables_secrets:
    HF_TOKEN: "your-secret-path/hf_token"
```

### Compute Resources

```yaml
compute:
  gpus: 8           # Total GPUs across all nodes
  gpu_type: h100    # GPU type: a10, h100, etc.
```

### Multi-Node Training

For distributed training across multiple nodes:

```yaml
command: |-
  torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    your_script.py
```

## Resources

- [Serverless GPU Compute Documentation]([https://docs.databricks.com/](https://docs.databricks.com/aws/en/compute/serverless/gpu))
