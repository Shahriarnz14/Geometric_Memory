# Geometric Memory (LLM)

Geometric-memory experiments for graph-structured memorization and path inference.

## Core Workflow

1. Generate an in-weights dataset (`pretrain/train/test`) with the dataset builder.
2. Run a training recipe (`staged_full_path`, `mixed_full_path`, `staged_hardest_token`, `mixed_hardest_token`).

## Dataset Generation

Use:

```bash
python data/build_in_weights_datasets.py --help
```

Example (path-star, forward+backward edges):

```bash
python data/build_in_weights_datasets.py \
  --graph_type star \
  --star_degree 5000 \
  --star_subtree_degree 1 \
  --path_length 5 \
  --add_forward_edges \
  --add_backward_edges \
  --random_seed 0
```

This writes files under:

`data/datasets/in_weights_graphs/star_graphs_randomized/`

## Training

```bash
conda activate geometric_memory
python train_in_weights.py \
  --training_recipe staged_full_path \
  --graph_type star \
  --star_degree 5000 \
  --star_subtree_degree 1 \
  --path_length 5 \
  --add_forward_edges \
  --add_backward_edges \
  --enable_wandb \
  --wandb_mode offline
```

Default prefix behavior:
- Task tokens are included in prefixes by default (`[EDGE]` / `[PATH]`).
- Edge memorization drops the pause token by default.

Optional overrides:
- `--exclude_task_token_in_prefix` disables task tokens in prefixes.
- `--edge_memorization_include_pause_token` keeps pause in edge-memorization prefixes.

## Logging

- W&B is the experiment logger.
- Use `--enable_wandb` and choose `--wandb_mode` (`online`, `offline`, `disabled`).
