"""Reduce noisy stdout/stderr from Hugging Face / transformers / tqdm on import."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import threading
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


@contextlib.contextmanager
def silence_fd(reraise_with_buffer: bool = True):
    """Capture **OS-level** stdout (fd 1) and stderr (fd 2) for a noisy block.

    ``silence_stdio`` only redirects Python's ``sys.stdout`` / ``sys.stderr``
    objects, so any C / Rust extension that writes directly via
    ``write(1, ...)`` or ``write(2, ...)`` (huggingface_hub, the BERT loader
    inside sentence-transformers, the tokenizers Rust crate, ChromaDB's onnx
    runtime, ...) bypasses it and still leaks to the terminal — garbling the
    spinner during the first RAG model load.

    This context manager goes one level deeper:

    1. ``os.dup`` the original fd 1 / fd 2 so we can restore them later.
    2. Replace fd 1 / fd 2 with the write end of an ``os.pipe()``.
    3. A daemon thread drains each pipe into an in-memory ``BytesIO`` buffer.
    4. On normal exit the buffer is silently discarded.
    5. On exception the buffer (decoded best-effort) is appended to the saved
       stderr so genuine failures (network, auth, disk) remain visible.

    The Python-level redirect is layered on top so that anything the
    interpreter prints via ``print()`` is also captured — belt-and-braces.

    IMPORTANT: keep the wrapped block as small as possible. Anything that
    needs to render to the terminal (rich Live spinners, progress bars) must
    live **outside** this context.
    """
    saved_stdout_fd: int | None = None
    saved_stderr_fd: int | None = None
    pipes: list[tuple[int, int]] = []
    threads: list[threading.Thread] = []
    buffers: list[bytearray] = []

    def _drain(read_fd: int, sink: bytearray) -> None:
        try:
            while True:
                data = os.read(read_fd, 4096)
                if not data:
                    break
                sink.extend(data)
        except OSError:
            pass

    py_buf_out = io.StringIO()
    py_buf_err = io.StringIO()

    try:
        try:
            saved_stdout_fd = os.dup(1)
            saved_stderr_fd = os.dup(2)
        except OSError:
            saved_stdout_fd = None
            saved_stderr_fd = None

        if saved_stdout_fd is not None and saved_stderr_fd is not None:
            for target_fd in (1, 2):
                read_fd, write_fd = os.pipe()
                os.dup2(write_fd, target_fd)
                os.close(write_fd)
                buf = bytearray()
                t = threading.Thread(target=_drain, args=(read_fd, buf), daemon=True)
                t.start()
                pipes.append((read_fd, target_fd))
                threads.append(t)
                buffers.append(buf)

        try:
            with (
                contextlib.redirect_stdout(py_buf_out),
                contextlib.redirect_stderr(py_buf_err),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore")
                yield
        finally:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            try:
                sys.stderr.flush()
            except Exception:
                pass

            if saved_stdout_fd is not None:
                try:
                    os.dup2(saved_stdout_fd, 1)
                finally:
                    os.close(saved_stdout_fd)
            if saved_stderr_fd is not None:
                try:
                    os.dup2(saved_stderr_fd, 2)
                finally:
                    os.close(saved_stderr_fd)

            for read_fd, _ in pipes:
                try:
                    os.close(read_fd)
                except OSError:
                    pass
            for t in threads:
                t.join(timeout=0.5)
    except Exception:
        if reraise_with_buffer:
            chunks: list[str] = []
            for buf in buffers:
                if buf:
                    chunks.append(buf.decode("utf-8", errors="replace"))
            py_extra = (py_buf_out.getvalue() + py_buf_err.getvalue()).strip()
            if py_extra:
                chunks.append(py_extra)
            captured = "\n".join(c.strip() for c in chunks if c.strip()).strip()
            if captured:
                try:
                    sys.stderr.write(captured + "\n")
                except Exception:
                    pass
        raise
