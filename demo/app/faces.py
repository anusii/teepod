"""Face detection and recognition with InsightFace's buffalo_l models (see models.py).

  det_10g.onnx    SCRFD-10GF: finds faces and five landmarks (eyes, nose, mouth corners)
  w600k_r50.onnx  ArcFace R50: turns each aligned face into a 512-d unit vector

Pre- and post-processing follow InsightFace's reference code (scrfd.py,
arcface_onnx.py, face_align.py), so the embeddings are compatible with other
buffalo_l users such as Immich. Two faces belong to the same person when the
cosine similarity of their embeddings is high (typically above ~0.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

# Where ArcFace expects the five landmarks in its 112x112 input.
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


@dataclass
class Face:
    box: np.ndarray        # x1, y1, x2, y2 in image pixels
    score: float           # detection confidence
    landmarks: np.ndarray  # 5 x 2, image pixels
    embedding: np.ndarray  # 512, L2-normalised


def make_session(path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    # The recogniser's metadata declares batch size 1 but it batches correctly;
    # hide ORT's per-call shape warning so it does not flood the logs.
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares rotation + uniform scale + translation (Umeyama), the same
    estimate as skimage's SimilarityTransform used by InsightFace. Returns 2x3."""
    n = src.shape[0]
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / n
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    u, s, vt = np.linalg.svd(cov)
    rot = u @ np.diag(d) @ vt
    scale = (s * d).sum() / (src_c ** 2).sum(axis=1).mean()
    m = np.zeros((2, 3), dtype=np.float32)
    m[:, :2] = scale * rot
    m[:, 2] = dst_mean - scale * rot @ src_mean
    return m


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return keep


class FaceAnalyzer:
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, detector_path, recogniser_path, threads: int = 1,
                 det_size: int = 640, det_thresh: float = 0.5, nms_thresh: float = 0.4):
        self.det = make_session(detector_path, threads)
        self.rec = make_session(recogniser_path, threads)
        self.det_input = self.det.get_inputs()[0].name
        self.rec_input = self.rec.get_inputs()[0].name
        self.det_size, self.det_thresh, self.nms_thresh = det_size, det_thresh, nms_thresh
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._centers:
            grid = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape(-1, 2)
            self._centers[key] = np.repeat(centers, self.NUM_ANCHORS, axis=0)
        return self._centers[key]

    def detect(self, rgb: np.ndarray, max_faces: int = 0):
        """Returns (boxes N x 4, scores N, landmarks N x 5 x 2), largest faces first."""
        h, w = rgb.shape[:2]
        size = self.det_size
        if h > w:
            new_h, new_w = size, max(1, int(size * w / h))
        else:
            new_w, new_h = size, max(1, int(size * h / w))
        scale = new_h / h
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(rgb, (new_w, new_h))
        blob = ((canvas.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.det.run(None, {self.det_input: blob})

        all_boxes, all_scores, all_kps = [], [], []
        n = len(self.STRIDES)
        for i, stride in enumerate(self.STRIDES):
            scores = outs[i].reshape(-1)
            deltas = outs[i + n].reshape(-1, 4) * stride
            kps = outs[i + 2 * n].reshape(-1, 10) * stride
            centers = self._anchor_centers(size // stride, size // stride, stride)
            keep = scores >= self.det_thresh
            if not keep.any():
                continue
            c, d, k = centers[keep], deltas[keep], kps[keep]
            all_boxes.append(np.concatenate([c - d[:, :2], c + d[:, 2:]], axis=1))
            all_scores.append(scores[keep])
            all_kps.append((np.tile(c, 5) + k).reshape(-1, 5, 2))
        if not all_scores:
            return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 5, 2))

        boxes = np.concatenate(all_boxes) / scale
        scores = np.concatenate(all_scores)
        kps = np.concatenate(all_kps) / scale
        keep = _nms(boxes, scores, self.nms_thresh)
        boxes, scores, kps = boxes[keep], scores[keep], kps[keep]

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        if max_faces:
            order = order[:max_faces]
        return boxes[order], scores[order], kps[order]

    def embed(self, rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Aligns each face to ArcFace's template and returns N x 512 unit vectors."""
        if len(landmarks) == 0:
            return np.zeros((0, 512), dtype=np.float32)
        crops = [cv2.warpAffine(rgb, similarity_transform(lm, ARCFACE_DST), (112, 112),
                                borderValue=0.0) for lm in landmarks]
        blob = ((np.stack(crops).astype(np.float32) - 127.5) / 127.5).transpose(0, 3, 1, 2)
        emb = self.rec.run(None, {self.rec_input: blob})[0]
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def analyze(self, rgb: np.ndarray, max_faces: int = 0) -> list[Face]:
        boxes, scores, kps = self.detect(rgb, max_faces)
        if len(boxes) == 0:
            # SCRFD misses faces that fill the whole frame (ID photos, avatars,
            # already-cropped faces). Retry once with a black border around it.
            pad = max(rgb.shape[:2]) // 2
            padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)))
            boxes, scores, kps = self.detect(padded, max_faces)
            boxes, kps = boxes - pad, kps - pad
            h, w = rgb.shape[:2]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            rgb = padded
            embeddings = self.embed(rgb, kps + pad)
        else:
            embeddings = self.embed(rgb, kps)
        return [Face(b, float(s), k, e) for b, s, k, e in zip(boxes, scores, kps, embeddings)]
