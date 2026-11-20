"""Zero-shot image tagging with Google's SigLIP ViT-B/16 (WebLI, 224 px), using
the ONNX export Immich publishes (see models.py).

SigLIP maps images and text into the same 768-d space. Tagging is done in two
stages:

  build time  `embed_tags` runs the TEXT half once over the tag vocabulary
              (tags.txt, each wrapped in PROMPT) and saves the unit vectors to
              tags.npz. The text model is then discarded and does not ship in
              the runtime image.
  runtime     `Tagger` runs only the IMAGE half and scores every tag by cosine
              similarity between the image vector and the tag's text vector.

Scores are raw cosine similarities, not probabilities: for SigLIP they are
small numbers (roughly -0.1 to 0.2), and only their ranking is meaningful.
"""

from __future__ import annotations

import string
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 224
CONTEXT_LENGTH = 64           # SigLIP's text length; it was trained with max-length padding
PROMPT = "a photo of {}"


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """SigLIP's training preprocessing: squash-resize to 224x224 (no crop, so
    content at the edges is kept), bicubic, then scale pixels to [-1, 1]."""
    img = Image.fromarray(rgb).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return x.transpose(2, 0, 1)


def canonicalize(text: str) -> str:
    """The text cleaning SigLIP's WebLI tokenizer expects: lower case, no
    punctuation, single spaces."""
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    return " ".join(text.split())


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class Tagger:
    def __init__(self, visual_path, tags_path, threads: int = 1):
        from faces import make_session
        self.session = make_session(visual_path, threads)
        self.input_name = self.session.get_inputs()[0].name
        with np.load(tags_path, allow_pickle=False) as data:
            self.names = [str(n) for n in data["names"]]
            self.vectors = data["embeddings"].astype(np.float32)
        probe = self.session.run(None, {self.input_name: np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), np.float32)})[0]
        if probe.shape[-1] != self.vectors.shape[1]:
            raise ValueError(f"tag vectors have {self.vectors.shape[1]} dims but the image model "
                             f"outputs {probe.shape[-1]}; rebuild tags.npz")

    def tag(self, rgb: np.ndarray, top_k: int = 5, min_score: float | None = None):
        """Returns [(tag, score)], best first."""
        x = preprocess(rgb)[None]
        image_vec = _unit(self.session.run(None, {self.input_name: x})[0][0])
        scores = self.vectors @ image_vec
        best = np.argsort(-scores)[:top_k]
        return [(self.names[i], float(scores[i])) for i in best
                if min_score is None or scores[i] >= min_score]


def load_vocabulary(path: Path) -> list[str]:
    tags, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        tag = line.split("#", 1)[0].strip()
        if tag and tag not in seen:
            tags.append(tag)
            seen.add(tag)
    if not tags:
        raise ValueError(f"{path} contains no tags")
    return tags


def embed_tags(textual_path, tokenizer_path, vocabulary: list[str], out_path, threads: int = 1):
    """Build-time step: embed every tag with SigLIP's text model and save tags.npz."""
    from tokenizers import Tokenizer

    from faces import make_session

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad = tokenizer.padding or {}
    pad_token = pad.get("pad_token", "</s>")
    pad_id = pad.get("pad_id", tokenizer.token_to_id(pad_token) or 1)
    tokenizer.enable_padding(length=CONTEXT_LENGTH, pad_id=pad_id, pad_token=pad_token)
    tokenizer.enable_truncation(max_length=CONTEXT_LENGTH)

    session = make_session(textual_path, threads)
    inp = session.get_inputs()[0]
    dtype = np.int64 if "int64" in inp.type else np.int32

    vectors = []
    for tag in vocabulary:
        ids = tokenizer.encode(canonicalize(PROMPT.format(tag))).ids
        vectors.append(session.run(None, {inp.name: np.array([ids], dtype=dtype)})[0][0])
    vectors = _unit(np.stack(vectors).astype(np.float32))
    np.savez(out_path, names=np.array(vocabulary), embeddings=vectors, prompt=np.array(PROMPT))
    return vectors.shape
