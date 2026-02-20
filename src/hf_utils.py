# src/hf_utils.py

from __future__ import annotations
import os
from typing import Iterable, Optional, Sequence
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError

__all__ = ["hf_download", "hf_download_robust"]

def hf_download(
    repo_id: str,
    filename: str,
    repo_type: str = "model",
    token: Optional[str] = None,
    force_download: bool = False,
    local_dir: Optional[str] = None,
) -> str:
    """
    Download a single file from Hugging Face Hub, returning the *local cached path*.
    Works with public or private repos (provide `token` for private).

    Parameters
    ----------
    repo_id : str
        E.g., "sivaramakrishhnan/cxr-dpn68-tb-cls"
    filename : str
        Exact file name in repo (supports subfolders), e.g., "dpn68_fold2.ckpt"
    repo_type : str
        "model" (default), "dataset", etc.
    token : Optional[str]
        HF token for private repos. Public repos can pass None.
    force_download : bool
        If True, re-download even if present in local cache.
    local_dir : Optional[str]
        If provided, cache into this directory instead of default HF cache.

    Returns
    -------
    str
        Absolute local path to the downloaded file.
    """
    # Avoid symlink warnings in some managed environments
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=token,
        force_download=force_download,
        local_dir=local_dir,
        local_dir_use_symlinks=True,
    )
    return path


def hf_download_robust(
    repo_id: str,
    filename: str,
    *,
    alt_filenames: Optional[Sequence[str]] = None,
    repo_type: str = "model",
    token: Optional[str] = None,
    force_download: bool = False,
    local_dir: Optional[str] = None,
) -> str:
    
    candidates: Iterable[str] = (filename,)
    if alt_filenames:
        candidates = tuple(filename for filename in (filename, *alt_filenames))

    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            return hf_download(
                repo_id=repo_id,
                filename=name,
                repo_type=repo_type,
                token=token,
                force_download=force_download,
                local_dir=local_dir,
            )
        except EntryNotFoundError as e:
            last_err = e
            continue

    # Snapshot fallback (only if all direct attempts failed)
    # Download a local snapshot, then return the path to the requested filename if present.
    try:
        repo_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            local_dir=local_dir,
            local_dir_use_symlinks=True,
            allow_patterns=None,  # whole repo
        )
        target_names = list(candidates)
        for name in target_names:
            cand = os.path.join(repo_path, name)
            if os.path.exists(cand):
                return cand
        # Some users put files at repo root under different capitalization
        # Try a loose scan if still not found:
        for root, _, files in os.walk(repo_path):
            for f in files:
                if os.path.basename(f).lower() in {n.lower() for n in target_names}:
                    return os.path.join(root, f)
    except Exception as e2:
        last_err = last_err or e2

    # If we reach here, we truly couldn't locate the file.
    raise last_err if last_err is not None else FileNotFoundError(
        f"Could not locate '{filename}' (or any alternates) in repo {repo_id}."
    )

