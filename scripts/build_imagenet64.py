#!/usr/bin/env python3
"""
Stream-download ILSVRC/imagenet-1k in small batches, resize all images to 64x64 RGB
using the standard ADM/DiT anti-aliased center-crop pipeline (lossless PNG),
and upload to Forbu14/imagenet-64 with full 1,000-class ClassLabel metadata preserved.

Peak local disk usage is < 2.5 GB at all times. Fully resumable across restarts.
"""

import argparse
import io
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple

import huggingface_hub
from huggingface_hub import HfApi, hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SRC_REPO = "ILSVRC/imagenet-1k"
DST_REPO = "Forbu14/imagenet-64"


def resize_image_64(item: Tuple[bytes, str]) -> dict:
    """
    ADM / DiT standard center_crop_arr(64):
      1. libjpeg DCT draft downsample so min(w, h) >= 128
      2. Convert to RGB
      3. Iteratively 2x BOX downsample while min(w, h) >= 128 (anti-aliased)
      4. LANCZOS resize shorter side to 64
      5. Center crop to 64x64 and encode as lossless PNG
    """
    img_bytes, path = item
    with Image.open(io.BytesIO(img_bytes)) as img:
        img.draft("RGB", (128, 128))
        img = img.convert("RGB")
        w, h = img.size
        while min(w, h) >= 128:
            img = img.resize((w // 2, h // 2), resample=Image.Resampling.BOX)
            w, h = img.size
        if min(w, h) != 64:
            scale = 64.0 / min(w, h)
            new_w = max(64, int(round(w * scale)))
            new_h = max(64, int(round(h * scale)))
            img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            w, h = img.size
        left = (w - 64) // 2
        top = (h - 64) // 2
        if w != 64 or h != 64:
            img = img.crop((left, top, left + 64, top + 64))
        out = io.BytesIO()
        img.save(out, format="PNG", compress_level=1)
        new_path = (path.rsplit(".", 1)[0] + ".png") if path else None
        return {"bytes": out.getvalue(), "path": new_path}


def make_batches(files: List[str], num_output_shards: int) -> List[Tuple[int, int, List[str]]]:
    """Partition `files` into `num_output_shards` contiguous groups."""
    n = len(files)
    batches = []
    for i in range(num_output_shards):
        start = (i * n) // num_output_shards
        end = ((i + 1) * n) // num_output_shards
        batches.append((i, num_output_shards, files[start:end]))
    return batches


def sync_metadata_files(api: HfApi):
    """Copy classes.py and construct the README.md dataset card with all 1,000 class labels."""
    existing = set(api.list_repo_files(DST_REPO, repo_type="dataset"))
    with tempfile.TemporaryDirectory() as tmp:
        if "classes.py" not in existing:
            classes_path = hf_hub_download(SRC_REPO, "classes.py", repo_type="dataset", local_dir=tmp)
            api.upload_file(
                path_or_fileobj=classes_path,
                path_in_repo="classes.py",
                repo_id=DST_REPO,
                repo_type="dataset",
                commit_message="Add ImageNet-1k classes.py mapping",
            )
            print("[metadata] Uploaded classes.py", flush=True)

        src_readme = hf_hub_download(SRC_REPO, "README.md", repo_type="dataset", local_dir=tmp)
        txt = open(src_readme, "r", encoding="utf-8").read()
        parts = txt.split("---")
        yaml_header = parts[1]
        # Strip gated prompt and old byte sizes from YAML while keeping the full 1000 class_label dictionary
        yaml_lines = yaml_header.splitlines()
        cleaned_lines = []
        skip_block = False
        for line in yaml_lines:
            if line.startswith(("extra_gated_prompt:", "  splits:", "  download_size:", "  dataset_size:", "  - split: test")):
                skip_block = True
                continue
            if skip_block:
                if line.startswith(("dataset_info:", "configs:")):
                    skip_block = False
                else:
                    continue
            cleaned_lines.append(line)

        new_readme = (
            "---\n"
            + "\n".join(cleaned_lines)
            + "\n---\n\n"
            + "# ImageNet-64 (True ILSVRC2012 Labels, Anti-Aliased 64×64 Center Crop)\n\n"
            + "Built directly from [`ILSVRC/imagenet-1k`](https://huggingface.co/datasets/ILSVRC/imagenet-1k) "
            + "preserving the exact original `label` (`0..999`) and `ClassLabel` metadata for all:\n"
            + "- **Validation:** `50,000` images (`data/validation-00000-of-00002.parquet` .. `00001`)\n"
            + "- **Train:** `1,281,167` images (`data/train-00000-of-00030.parquet` .. `00029`)\n\n"
            + "## Preprocessing (ADM / DiT `center_crop_arr(64)`)\n"
            + "1. Iterative $2\\times$ `Image.BOX` downsampling while $\\min(W, H) \\ge 128$ to prevent aliasing.\n"
            + "2. `Image.LANCZOS` resize so $\\min(W, H) = 64$.\n"
            + "3. Exact $64 \\times 64$ center crop encoded as **lossless RGB PNG**.\n"
        )
        readme_out = os.path.join(tmp, "README_out.md")
        with open(readme_out, "w", encoding="utf-8") as f:
            f.write(new_readme)
        api.upload_file(
            path_or_fileobj=readme_out,
            path_in_repo="README.md",
            repo_id=DST_REPO,
            repo_type="dataset",
            commit_message="Update README.md with ImageNet-1k ClassLabel metadata",
        )
        print("[metadata] Uploaded README.md with 1000 ClassLabel names", flush=True)


def download_shard_with_retry(filename: str, dst_dir: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            path = hf_hub_download(
                repo_id=SRC_REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=dst_dir,
            )
            return path
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [warn] Download failed for {filename} (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {filename} after {max_retries} attempts")


def upload_shard_with_retry(api: HfApi, local_path: str, repo_path: str, msg: str, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=DST_REPO,
                repo_type="dataset",
                commit_message=msg,
            )
            return
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  [warn] Upload failed for {repo_path} (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Failed to upload {repo_path} after {max_retries} attempts")


def process_split(
    api: HfApi,
    split_name: str,
    src_files: List[str],
    num_output_shards: int,
    workers: int = 8,
):
    existing = set(api.list_repo_files(DST_REPO, repo_type="dataset"))
    batches = make_batches(src_files, num_output_shards)

    # Filter to remaining batches
    todo_batches = []
    for shard_idx, total_shards, batch_files in batches:
        out_repo_path = f"data/{split_name}-{shard_idx:05d}-of-{total_shards:05d}.parquet"
        if out_repo_path in existing:
            print(f"[{split_name}] Skipping already uploaded {out_repo_path}", flush=True)
        else:
            todo_batches.append((shard_idx, total_shards, batch_files, out_repo_path))

    if not todo_batches:
        print(f"[{split_name}] All {num_output_shards} shards already uploaded!", flush=True)
        return

    print(
        f"[{split_name}] Processing {len(todo_batches)}/{num_output_shards} output shards "
        f"({sum(len(b[2]) for b in todo_batches)} source files)...",
        flush=True,
    )

    # Background uploader thread so next batch can start downloading/resizing while previous uploads
    upload_queue: "queue.Queue[Tuple[str, str, str, str] | None]" = queue.Queue(maxsize=2)
    upload_error = []

    def uploader_loop():
        while True:
            item = upload_queue.get()
            if item is None:
                upload_queue.task_done()
                break
            local_pq, repo_pq, commit_msg, tmp_dir_to_clean = item
            try:
                t_up = time.time()
                sz_mb = os.path.getsize(local_pq) / 1e6
                upload_shard_with_retry(api, local_pq, repo_pq, commit_msg)
                dt_up = time.time() - t_up
                print(
                    f"  [upload done] {repo_pq} ({sz_mb:.1f} MB in {dt_up:.1f}s, {sz_mb/max(dt_up, 1e-3):.1f} MB/s)",
                    flush=True,
                )
            except Exception as e:
                upload_error.append(e)
            finally:
                shutil.rmtree(tmp_dir_to_clean, ignore_errors=True)
                upload_queue.task_done()

    up_thread = threading.Thread(target=uploader_loop, daemon=True)
    up_thread.start()

    t_split_start = time.time()
    total_imgs_done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for b_pos, (shard_idx, total_shards, batch_files, out_repo_path) in enumerate(todo_batches):
            if upload_error:
                raise upload_error[0]

            t_batch = time.time()
            batch_tmp = tempfile.mkdtemp(prefix=f"in64_{split_name}_{shard_idx:03d}_")

            # Producer thread: strictly bounded to at most 2 downloaded source shards on disk at once
            dl_queue: "queue.Queue[Tuple[str, str, str] | Exception | None]" = queue.Queue(maxsize=2)

            def producer(files_to_dl: List[str], tmp_root: str):
                try:
                    for f_idx, fname in enumerate(files_to_dl):
                        sub_tmp = os.path.join(tmp_root, f"dl_{f_idx}")
                        os.makedirs(sub_tmp, exist_ok=True)
                        local_p = download_shard_with_retry(fname, sub_tmp)
                        dl_queue.put((fname, sub_tmp, local_p))
                    dl_queue.put(None)
                except Exception as e:
                    dl_queue.put(e)

            prod_t = threading.Thread(target=producer, args=(batch_files, batch_tmp), daemon=True)
            prod_t.start()

            out_tables = []
            schema_ref = None
            batch_imgs = 0

            while True:
                item = dl_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    shutil.rmtree(batch_tmp, ignore_errors=True)
                    raise item
                fname, sub_tmp, local_p = item
                table = pq.read_table(local_p)
                if schema_ref is None:
                    schema_ref = table.schema
                raw_images = table["image"].to_pylist()
                labels_arr = table["label"]
                # Immediately delete the 500MB raw parquet shard to keep disk usage tiny
                shutil.rmtree(sub_tmp, ignore_errors=True)

                work_items = [(x["bytes"], x["path"]) for x in raw_images]
                del raw_images
                resized_images = list(pool.map(resize_image_64, work_items, chunksize=64))
                del work_items

                sub_table = pa.Table.from_arrays(
                    [pa.array(resized_images, type=schema_ref.field("image").type), labels_arr],
                    schema=schema_ref,
                )
                out_tables.append(sub_table)
                batch_imgs += len(resized_images)
                del resized_images

            combined_table = pa.concat_tables(out_tables)
            del out_tables

            out_local_pq = os.path.join(batch_tmp, os.path.basename(out_repo_path))
            pq.write_table(combined_table, out_local_pq, compression="zstd")
            del combined_table

            total_imgs_done += batch_imgs
            dt_batch = time.time() - t_batch
            elapsed_total = time.time() - t_split_start
            out_mb = os.path.getsize(out_local_pq) / 1e6
            print(
                f"[{split_name} {shard_idx+1:02d}/{total_shards:02d}] "
                f"Prepared {batch_imgs:,} imgs ({len(batch_files)} src shards -> {out_mb:.1f} MB) "
                f"in {dt_batch:.1f}s ({batch_imgs/max(dt_batch,1e-3):.0f} img/s) | "
                f"Total: {total_imgs_done:,} imgs in {elapsed_total/60:.1f}m",
                flush=True,
            )

            commit_msg = f"Add {out_repo_path} ({batch_imgs} 64x64 images from {len(batch_files)} source shards)"
            upload_queue.put((out_local_pq, out_repo_path, commit_msg, batch_tmp))

    upload_queue.put(None)
    up_thread.join()
    if upload_error:
        raise upload_error[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-shards", type=int, default=2)
    parser.add_argument("--train-shards", type=int, default=30)
    parser.add_argument("--split", choices=["all", "validation", "train"], default="all")
    args = parser.parse_args()

    api = HfApi()
    print(f"Authenticated as: {api.whoami()['name']}", flush=True)
    sync_metadata_files(api)

    info = api.dataset_info(SRC_REPO)
    val_files = sorted(
        s.rfilename for s in info.siblings if s.rfilename.startswith("data/validation-") and s.rfilename.endswith(".parquet")
    )
    train_files = sorted(
        s.rfilename for s in info.siblings if s.rfilename.startswith("data/train-") and s.rfilename.endswith(".parquet")
    )
    print(f"Found {len(val_files)} validation shards and {len(train_files)} train shards in {SRC_REPO}", flush=True)

    if args.split in ("all", "validation"):
        process_split(api, "validation", val_files, args.val_shards, workers=args.workers)
    if args.split in ("all", "train"):
        process_split(api, "train", train_files, args.train_shards, workers=args.workers)

    print("All requested splits processed and uploaded to Forbu14/imagenet-64!", flush=True)


if __name__ == "__main__":
    main()
