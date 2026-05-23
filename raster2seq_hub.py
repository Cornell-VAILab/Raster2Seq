"""Utilities for downloading Raster2Seq checkpoints from Hugging Face Hub."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional


DEFAULT_REPO_ID = "haopt/Raster2Seq"
CHECKPOINT_FILENAME = "checkpoint.pth"

CHECKPOINTS: Dict[str, Dict[str, object]] = {
    "s3d-bw": {
        "name": "Structured3D-B",
        "subfolder": "s3d-bw",
        "local_filename": "s3dbw_sem_res256_ep0449.pth",
        "room_f1": 99.6,
    },
    "cubicasa5k": {
        "name": "CubiCasa5K",
        "subfolder": "cubicasa5k",
        "local_filename": "cc5k_sem_res256_ep0499.pth",
        "room_f1": 88.7,
    },
    "raster2graph": {
        "name": "Raster2Graph",
        "subfolder": "raster2graph",
        "local_filename": "r2g_sem_res256_ep0549.pth",
        "room_f1": 97.0,
    },
    "raster2graph-512": {
        "name": "Raster2Graph-512",
        "subfolder": "Raster2Graph-512",
        "local_filename": "r2g_sem_res512_ep0749.pth",
        "room_f1": 97.0,
    },
    "s3d-density": {
        "name": "Structured3D-DensityMap",
        "subfolder": "s3d-density",
        "local_filename": "s3dd_sem_res256_ep0699.pth",
        "room_f1": 99.1,
    },
}

ALIASES = {
    "s3d": "s3d-bw",
    "structured3d": "s3d-bw",
    "structured3d-b": "s3d-bw",
    "s3dbw": "s3d-bw",
    "cc5k": "cubicasa5k",
    "cubicasa": "cubicasa5k",
    "r2g": "raster2graph",
    "r2g-512": "raster2graph-512",
    "raster2graph512": "raster2graph-512",
    "s3dd": "s3d-density",
    "structured3d-density": "s3d-density",
}


def normalize_checkpoint_name(name: str) -> str:
    """Return the canonical checkpoint key for a user-facing name or alias."""
    normalized = name.strip().lower().replace("_", "-")
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in CHECKPOINTS:
        valid = ", ".join(sorted(CHECKPOINTS))
        aliases = ", ".join(sorted(ALIASES))
        raise ValueError(
            f"Unknown Raster2Seq checkpoint '{name}'. Valid checkpoints: {valid}. "
            f"Accepted aliases: {aliases}."
        )
    return normalized


def _import_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Downloading Raster2Seq checkpoints from Hugging Face requires "
            "`huggingface_hub`. Install it with `pip install huggingface_hub`."
        ) from exc
    return hf_hub_download, snapshot_download


def download_checkpoint(
    checkpoint: str,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = CHECKPOINT_FILENAME,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Download one checkpoint subfolder and return the cached checkpoint path."""
    checkpoint_key = normalize_checkpoint_name(checkpoint)
    subfolder = str(CHECKPOINTS[checkpoint_key]["subfolder"])
    local_repo = download_checkpoint_folder(
        checkpoint_key,
        repo_id=repo_id,
        cache_dir=cache_dir,
        revision=revision,
    )
    return str(Path(local_repo) / subfolder / filename)


def download_all_checkpoints(
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Download the full Raster2Seq checkpoint repository and return its local path."""
    _, snapshot_download = _import_huggingface_hub()
    return snapshot_download(repo_id=repo_id, cache_dir=cache_dir, revision=revision)


def download_checkpoint_folder(
    checkpoint: str,
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Download all files for one checkpoint subfolder and return the local repo path."""
    _, snapshot_download = _import_huggingface_hub()
    checkpoint_key = normalize_checkpoint_name(checkpoint)
    subfolder = str(CHECKPOINTS[checkpoint_key]["subfolder"])
    return snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{subfolder}/*",
        cache_dir=cache_dir,
        revision=revision,
    )


def download_config(
    checkpoint: str,
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Download the config.json for one Raster2Seq checkpoint."""
    hf_hub_download, _ = _import_huggingface_hub()
    checkpoint_key = normalize_checkpoint_name(checkpoint)
    return hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        subfolder=str(CHECKPOINTS[checkpoint_key]["subfolder"]),
        cache_dir=cache_dir,
        revision=revision,
    )


def load_config(
    checkpoint: str,
    repo_id: str = DEFAULT_REPO_ID,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> Dict[str, object]:
    """Download and read the config.json for one Raster2Seq checkpoint."""
    config_path = Path(download_config(checkpoint, repo_id=repo_id, cache_dir=cache_dir, revision=revision))
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint_path(checkpoint: str) -> str:
    """Resolve a local checkpoint path or an `hf:<name>` Raster2Seq Hub alias."""
    if checkpoint.startswith("hf:"):
        return download_checkpoint(checkpoint.split(":", 1)[1])
    return checkpoint


download_raster2seq_checkpoint = download_checkpoint
