"""Reduce noisy stdout/stderr from Hugging Face / transformers / tqdm on import."""

from __future__ import annotations

import logging
import os


def suppress_hf_transformers_noise() -> None:
    """Tune env + log levels before loading sentence-transformers / torch."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Hugging Face Hub: hide tqdm bars when downloading/caching models
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # Avoid hanging forever on slow or blocked networks (seconds; hub default is often low)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    # Transformers 4.40+: less chatter
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    for name in (
        "transformers",
        "transformers.modeling_utils",
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "huggingface_hub",
        "huggingface_hub.file_download",
        "torch",
        "chromadb",
        "chromadb.telemetry",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
