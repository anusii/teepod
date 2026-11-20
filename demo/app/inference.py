"""Image decoding and the inference engine shared by all client streams.

Both services run on ONNX Runtime in a small worker pool, so the asyncio event
loop stays free to accept streams while the CPU is busy. Each worker runs one
image at a time (the face detector's export takes a single image), and ONNX
Runtime parallelises inside each model with `threads` CPU threads, so workers
x threads matches the vCPUs.
"""

import asyncio
import io
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import models
from faces import FaceAnalyzer
from tagger import Tagger

log = logging.getLogger(__name__)

# Treat decompression bombs as hard errors; downscale huge photos before analysis.
Image.MAX_IMAGE_PIXELS = 80_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)
MAX_SIDE = int(os.environ.get("MAX_SIDE", "4096"))

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))
CPUS = os.cpu_count() or 1
WORKERS = int(os.environ.get("INFERENCE_WORKERS", str(max(1, CPUS // 2))))
THREADS = max(1, CPUS // WORKERS)


def decode_image(data: bytes) -> tuple[np.ndarray, float]:
    """Returns (RGB array, factor to scale coordinates back to the oriented original)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # phone photos store rotation in EXIF
    img = img.convert("RGB")
    factor = 1.0
    if max(img.size) > MAX_SIDE:
        factor = max(img.size) / MAX_SIDE
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return np.asarray(img), factor


class InferenceEngine:
    def __init__(self, faces: FaceAnalyzer, tagger: Tagger, workers: int = WORKERS):
        self.faces = faces
        self.tagger = tagger
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="infer")

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR):
        """Verifies every model against the digests pinned in models.py (fail
        closed: never load a damaged or substituted file), then loads them."""
        det, rec = models.ensure_all(model_dir, download=False)
        (visual,) = models.ensure_siglip(model_dir, (models.TAG_IMAGE_MODEL,), download=False)
        log.info("models verified; %d inference workers x %d threads", WORKERS, THREADS)
        return cls(FaceAnalyzer(det, rec, THREADS), Tagger(visual, model_dir / "tags.npz", THREADS))

    async def _run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    def _faces(self, data: bytes, max_faces: int):
        rgb, factor = decode_image(data)
        found = self.faces.analyze(rgb, max_faces)
        for f in found:
            f.box, f.landmarks = f.box * factor, f.landmarks * factor
        h, w = rgb.shape[:2]
        return found, round(w * factor), round(h * factor)

    def _tags(self, data: bytes, top_k: int, min_score):
        rgb, _ = decode_image(data)
        return self.tagger.tag(rgb, top_k, min_score)

    async def recognise_faces(self, data: bytes, max_faces: int):
        return await self._run(self._faces, data, max_faces)

    async def tag_image(self, data: bytes, top_k: int, min_score=None):
        return await self._run(self._tags, data, top_k, min_score)

    async def stop(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
