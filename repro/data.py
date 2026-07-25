"""UCF-101 download + preprocessing to fixed 16x64x64 uint8 clip tensors.

Fixed public subset: official UCF-101 (via HF mirror quchenyuan/UCF101-ZIP),
official recognition split 1. Train: up to 4 evenly spaced clips/video.
Held-out: test split 1, up to 2 clips/video. Deterministic; cached on shared PVC.
"""

import json
import multiprocessing as mp
import os
import time
import zipfile

import numpy as np

from . import constants as C

MARKER = os.path.join(C.DATA_DIR, f"done_{C.DATA_VERSION}.json")


def _decode_video(path):
    import av

    frames = []
    try:
        with av.open(path) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
    except Exception:
        pass
    return frames


def _resize_frames(frames):
    from PIL import Image

    out = []
    for f in frames:
        h, w = f.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = Image.fromarray(f[y0 : y0 + s, x0 : x0 + s]).resize((C.RES, C.RES), Image.BILINEAR)
        out.append(np.asarray(img))
    return np.stack(out)


def _extract_clips(args):
    path, label, n_clips = args
    frames = _decode_video(path)
    span = C.FRAMES * C.FRAME_STRIDE
    if len(frames) < span:
        return None
    starts = np.unique(np.linspace(0, len(frames) - span, n_clips).astype(int))
    clips = []
    for s in starts:
        clip = _resize_frames(frames[s : s + span : C.FRAME_STRIDE])
        clips.append(clip)
    return np.stack(clips).astype(np.uint8), label


def _prepare():
    from huggingface_hub import hf_hub_download

    os.makedirs(C.DATA_DIR, exist_ok=True)
    os.makedirs(C.HF_CACHE, exist_ok=True)
    t0 = time.time()
    zpath = hf_hub_download("quchenyuan/UCF101-ZIP", "UCF-101.zip", repo_type="dataset", cache_dir=C.HF_CACHE)
    spath = hf_hub_download(
        "quchenyuan/UCF101-ZIP", "UCF101TrainTestSplits-RecognitionTask.zip", repo_type="dataset", cache_dir=C.HF_CACHE
    )
    print(f"[data] downloads done in {time.time()-t0:.0f}s", flush=True)

    vid_dir = os.path.join(C.DATA_DIR, "videos")
    if not os.path.exists(os.path.join(vid_dir, "UCF-101")):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(vid_dir)
    split_dir = os.path.join(C.DATA_DIR, "splits")
    with zipfile.ZipFile(spath) as z:
        z.extractall(split_dir)
    root = os.path.join(vid_dir, "UCF-101")
    sd = os.path.join(split_dir, "ucfTrainTestlist")

    classes = {}
    with open(os.path.join(sd, "classInd.txt")) as f:
        for line in f:
            idx, name = line.split()
            classes[name] = int(idx) - 1

    def read_split(fname):
        items = []
        with open(os.path.join(sd, fname)) as f:
            for line in f:
                rel = line.split()[0].strip()
                cls = rel.split("/")[0]
                items.append((os.path.join(root, rel), classes[cls]))
        return items

    train_items = read_split("trainlist01.txt")
    test_items = read_split("testlist01.txt")
    print(f"[data] {len(train_items)} train videos, {len(test_items)} test videos", flush=True)

    for split, items, n_clips in [("train", train_items, C.TRAIN_CLIPS_PER_VIDEO), ("val", test_items, C.VAL_CLIPS_PER_VIDEO)]:
        t1 = time.time()
        args = [(p, l, n_clips) for p, l in items]
        clips_all, labels_all = [], []
        with mp.Pool(min(32, os.cpu_count())) as pool:
            for i, res in enumerate(pool.imap(_extract_clips, args, chunksize=8)):
                if res is not None:
                    clips, label = res
                    clips_all.append(clips)
                    labels_all.extend([label] * len(clips))
                if (i + 1) % 2000 == 0:
                    print(f"[data] {split}: {i+1}/{len(args)} videos, {time.time()-t1:.0f}s", flush=True)
        arr = np.concatenate(clips_all)
        labels = np.array(labels_all, dtype=np.int64)
        np.save(os.path.join(C.DATA_DIR, f"{split}_clips.npy"), arr)
        np.save(os.path.join(C.DATA_DIR, f"{split}_labels.npy"), labels)
        print(f"[data] {split}: {arr.shape} saved in {time.time()-t1:.0f}s", flush=True)

    with open(MARKER, "w") as f:
        json.dump({"version": C.DATA_VERSION, "time": time.time()}, f)


def ensure_data():
    """Idempotent, single-writer data prep. mkdir is atomic on NFS; non-owners poll the marker."""
    os.makedirs(C.DATA_DIR, exist_ok=True)
    if os.path.exists(MARKER):
        return
    lock_dir = os.path.join(C.DATA_DIR, "prep.lockdir")
    t0 = time.time()
    while not os.path.exists(MARKER):
        try:
            os.mkdir(lock_dir)  # atomic on NFS; re-attempted so a crashed owner's release is picked up
        except FileExistsError:
            if time.time() - t0 > 3 * 3600:
                raise RuntimeError("timed out waiting for data prep by another worker")
            print("[data] waiting for another worker's prep...", flush=True)
            time.sleep(15)
            continue
        try:
            _prepare()
        except BaseException:
            os.rmdir(lock_dir)
            raise
        break


def load_split(split):
    clips = np.load(os.path.join(C.DATA_DIR, f"{split}_clips.npy"))
    labels = np.load(os.path.join(C.DATA_DIR, f"{split}_labels.npy"))
    return clips, labels
