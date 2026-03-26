"""Generic experiment logging adapters and factories."""

from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


class ExperimentLogger:
    """Base logger interface used by training loops."""

    enabled = False

    def log_config(self, config: Dict) -> None:
        """Log config.
        
        Args:
            config: Input parameter.
        
        Returns:
            object: Function return value.
        """
        del config

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Input parameter.
            step: Input parameter.
        
        Returns:
            object: Function return value.
        """
        del metrics, step

    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """Log text.
        
        Args:
            key: Input parameter.
            text: Input parameter.
            step: Input parameter.
        
        Returns:
            object: Function return value.
        """
        del key, text, step

    def log_prediction_rows(self, key: str, rows: Sequence, step: Optional[int] = None) -> None:
        """Log prediction rows.
        
        Args:
            key: Input parameter.
            rows: Input parameter.
            step: Input parameter.
        
        Returns:
            object: Function return value.
        """
        del key, rows, step

    def finish(self) -> None:
        """Finish.
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        return None

    def get_run_url(self) -> Optional[str]:
        """Get run url.
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        return None

    def get_status_message(self) -> Optional[str]:
        """Get status message.
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        return None


@dataclass
class PredictionRow:
    """PredictionRow definition.
    
    Description:
        Encapsulates related behavior and state.
    """
    prefix: str
    target: str
    prediction: str


class NullExperimentLogger(ExperimentLogger):
    """No-op logger used when logging is disabled or unavailable."""

    enabled = False

    def __init__(self, reason: Optional[str] = None):
        """  init  .
        
        Args:
            reason: Input parameter.
        
        Returns:
            object: Function return value.
        """
        self._reason = reason

    def get_status_message(self) -> Optional[str]:
        """Get status message.
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        return self._reason


class WandbExperimentLogger(ExperimentLogger):
    """W&B-backed logger implementation."""

    enabled = True

    def __init__(
        self,
        *,
        project: str,
        name: str,
        mode: str,
        save_dir: str,
        config: Dict,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ):
        """  init  .
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        import wandb

        init_kwargs = {
            "project": project,
            "name": name,
            "mode": mode,
            "dir": save_dir,
            "config": config,
        }
        if entity:
            init_kwargs["entity"] = entity
        if group:
            init_kwargs["group"] = group
        if tags:
            init_kwargs["tags"] = list(tags)

        self._wandb = wandb
        self._run = wandb.init(**init_kwargs)
        self._mode = mode

    def log_config(self, config: Dict) -> None:
        """Log config.
        
        Args:
            config: Input parameter.
        
        Returns:
            object: Function return value.
        """
        self._run.config.update(config, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Input parameter.
            step: Input parameter.
        
        Returns:
            object: Function return value.
        """
        if not metrics:
            return
        self._run.log(metrics, step=step)

    def log_text(self, key: str, text: str, step: Optional[int] = None) -> None:
        """Log text.
        
        Args:
            key: Input parameter.
            text: Input parameter.
            step: Input parameter.
        
        Returns:
            object: Function return value.
        """
        self._run.log({key: self._wandb.Html(f"<pre>{html.escape(text)}</pre>")}, step=step)

    def log_prediction_rows(self, key: str, rows: Sequence, step: Optional[int] = None) -> None:
        """Log prediction rows.
        
        Args:
            key: Input parameter.
            rows: Input parameter.
            step: Input parameter.
        
        Returns:
            object: Function return value.
        """
        table = self._wandb.Table(columns=["prefix", "target", "prediction"])
        for row in rows:
            table.add_data(row.prefix, row.target, row.prediction)
        self._run.log({key: table}, step=step)

    def finish(self) -> None:
        """Finish.
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        self._run.finish()

    def get_run_url(self) -> Optional[str]:
        """Get run url.
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        url = getattr(self._run, "url", None)
        if not url and hasattr(self._run, "get_url"):
            try:
                url = self._run.get_url()
            except Exception:
                url = None
        if url:
            return url

        entity = getattr(self._run, "entity", None)
        project = getattr(self._run, "project", None)
        run_id = getattr(self._run, "id", None)
        base_url = os.getenv("WANDB_BASE_URL", "https://wandb.ai").rstrip("/")
        if entity and project and run_id:
            return f"{base_url}/{entity}/{project}/runs/{run_id}"
        return url

    def get_status_message(self) -> Optional[str]:
        """Get status message.
        
        Args:
            None: This callable does not take external parameters.
        
        Returns:
            object: Function return value.
        """
        if self._mode != "online":
            return f"W&B mode is '{self._mode}'. Cloud links require mode='online'."
        return None


def parse_comma_separated_tags(raw_tags: str) -> List[str]:
    """Parses comma-separated tags from CLI/config values.

    Args:
        raw_tags: Input parameter.

    Returns:
        object: Function return value.
    """
    if not raw_tags.strip():
        return []
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def create_experiment_logger(args, run_name: str) -> ExperimentLogger:
    """Creates an experiment logger from parsed args, with safe fallback.

    Args:
        args: Input parameter.
        run_name: Input parameter.

    Returns:
        object: Function return value.
    """
    if not getattr(args, "enable_wandb", False):
        return NullExperimentLogger(reason="W&B disabled by CLI flag.")

    try:
        tags = parse_comma_separated_tags(getattr(args, "wandb_tags", ""))
        return WandbExperimentLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group,
            tags=tags,
            mode=args.wandb_mode,
            save_dir=args.experiment_log_root,
            config=json.loads(json.dumps(vars(args), default=str)),
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(
            "Warning: failed to initialize Weights & Biases logging."
            f" Falling back to no-op logger. Error: {exc}"
        )
        return NullExperimentLogger(reason=f"W&B initialization failed: {exc}")


def make_prediction_rows(tokenizer, prediction_txt: Dict, max_rows: int = 20):
    """Converts prediction token ids to decoded rows for logging.

    Args:
        tokenizer: Input parameter.
        prediction_txt: Input parameter.
        max_rows: Input parameter.

    Returns:
        object: Function return value.
    """
    rows = []
    prefixes = prediction_txt.get("prefix", [])[:max_rows]
    targets = prediction_txt.get("target", [])[:max_rows]
    preds = prediction_txt.get("pred", [])[:max_rows]

    for prefix_tokens, target_tokens, pred_tokens in zip(prefixes, targets, preds):
        rows.append(
            PredictionRow(
                prefix=str(tokenizer.decode(prefix_tokens)),
                target=str(tokenizer.decode(target_tokens)),
                prediction=str(tokenizer.decode(pred_tokens)),
            )
        )
    return rows


def prediction_rows_to_markdown(rows: Iterable[PredictionRow]) -> str:
    """Renders prediction samples as a markdown table string.

    Args:
        rows: Input parameter.

    Returns:
        object: Function return value.
    """
    header = "**Prefix** | **Target** | **Prediction**\n---|---|---"
    lines = [header]
    for row in rows:
        lines.append(f"{row.prefix} | {row.target} | {row.prediction}")
    return "\n".join(lines)


def write_all_scalars_once(logger, scalar_dict, step):
    """Logs a scalar dictionary at a given step when a logger is available.

    Args:
        logger: Experiment logger implementing `log_metrics`.
        scalar_dict: Dictionary of scalar metrics to log.
        step: Global step or epoch index for the log event.

    Returns:
        None: Writes metrics through the logger when available.
    """
    if logger is None:
        return
    if hasattr(logger, "log_metrics"):
        logger.log_metrics(scalar_dict, step=step)


@dataclass(frozen=True)
class RunDirectories:
    """Filesystem layout for one experiment run.

    Args:
        run_dir: Root directory of the run.
        checkpoints_dir: Directory for checkpoints.
        artifacts_dir: Directory for misc artifacts.

    Returns:
        RunDirectories: Dataclass carrying resolved run paths.
    """

    run_dir: Path
    checkpoints_dir: Path
    artifacts_dir: Path


def prepare_run_directories(experiment_log_root: str, dataset: str, run_name: str):
    """Creates the standard run directory tree.

    Args:
        experiment_log_root: Root directory for all experiment logs.
        dataset: Dataset namespace used as a subfolder.
        run_name: Unique run name used as a subfolder.

    Returns:
        RunDirectories: Paths for run root, checkpoints, and artifacts.
    """
    run_dir = Path(experiment_log_root) / dataset / run_name
    checkpoints_dir = run_dir / "checkpoints"
    artifacts_dir = run_dir / "artifacts"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return RunDirectories(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        artifacts_dir=artifacts_dir,
    )
