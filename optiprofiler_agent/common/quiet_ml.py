"""Reduce noisy stdout/stderr from Hugging Face / transformers / tqdm on import."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import warnings


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


@contextlib.contextmanager
def silence_stdio(reraise_with_buffer: bool = True):
    """Capture stdout / stderr / warnings during a noisy block (e.g. model load).

    Many third-party libraries (huggingface_hub, sentence-transformers, the
    underlying BERT loader) print directly to stdout / stderr or use
    ``warnings.warn``. These messages bypass our logging configuration and
    overwhelm the spinner during the first RAG model load.

    On normal completion the captured text is silently discarded. If the
    wrapped block raises, the captured output is appended to the exception
    message (when ``reraise_with_buffer`` is True), so genuine failures
    (network, auth, disk) remain visible to the user.
    """
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(buf_out),
            contextlib.redirect_stderr(buf_err),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            yield
    except Exception as exc:
        if reraise_with_buffer:
            captured = (buf_out.getvalue() + buf_err.getvalue()).strip()
            if captured:
                sys.stderr.write(captured + "\n")
        raise exc
