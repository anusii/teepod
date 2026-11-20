"""Build-time model preparation, run by the Dockerfile's `models` stage.

  python prepare_models.py fetch      --models /models
      Fetches and SHA-256-verifies the face models (buffalo_l) and SigLIP's
      image half, using models.py.

  python prepare_models.py embed-tags --models /models --tags tags.txt
      Fetches SigLIP's text half and tokenizer (also verified), embeds every
      tag once and writes /models/tags.npz. Only the image half and tags.npz
      are copied into the runtime image.

Everything fetched is pinned by digest in models.py, so the runtime image -- and
therefore the attested image digest -- always contains exactly the same models.
"""

import argparse
import logging
import os
from pathlib import Path

import models
import tagger


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "embed-tags"])
    ap.add_argument("--models", type=Path, required=True, help="model folder")
    ap.add_argument("--tags", type=Path, default=Path("tags.txt"), help="tag vocabulary")
    args = ap.parse_args()

    if args.command == "fetch":
        det, rec = models.ensure_all(args.models)
        (visual,) = models.ensure_siglip(args.models, (models.TAG_IMAGE_MODEL,))
        print(f"verified: {det}, {rec}, {visual}")
    else:
        vocabulary = tagger.load_vocabulary(args.tags)
        textual, tok = models.ensure_siglip(args.models, (models.TAG_TEXT_MODEL, models.TAG_TOKENISER))
        out = args.models / "tags.npz"
        shape = tagger.embed_tags(textual, tok, vocabulary, out, threads=os.cpu_count() or 1)
        print(f"embedded {shape[0]} tags ({shape[1]} dims) into {out}")


if __name__ == "__main__":
    main()
