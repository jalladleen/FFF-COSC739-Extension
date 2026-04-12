"""Multi-GPU parallel training script for the generative canary ensemble.

Trains N generators by splitting them across all visible GPUs (one worker per GPU).
Each worker is a separate process with its own CUDA_VISIBLE_DEVICES, its own YOLOv8
detector instance, and its own in-memory dataset.

Usage:
    python train_ensemble_parallel.py
    python train_ensemble_parallel.py --n_generators 1000
    python train_ensemble_parallel.py --n_gpus 2
    python train_ensemble_parallel.py --checkpoint_dir ./ensemble_1000/checkpoints

Resume: skips any generator whose checkpoint already exists.
"""
import argparse
import os
import sys
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_generators', type=int, default=1000)
    p.add_argument('--n_gpus', type=int, default=0, help='0 = auto-detect')
    p.add_argument('--checkpoint_dir', type=str, default='./ensemble_1000/checkpoints')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight', type=float, default=2.0, help='lambda_c for FFF loss')
    p.add_argument('--z_dim', type=int, default=128)
    p.add_argument('--canary_cls_id', type=int, default=22)
    p.add_argument('--benign_root', type=str,
                   default='../Data/traineval/VOC07_YOLOv8/train_120/benign')
    p.add_argument('--benign_label_root', type=str,
                   default='../Data/traineval/VOC07_YOLOv8/train_120/benign_label')
    p.add_argument('--adversarial_root', type=str,
                   default='../Data/traineval/VOC07_YOLOv8/train_120/adversarial')
    p.add_argument('--img_size', type=int, default=640)
    return p.parse_args()


def train_worker(gpu_id, gen_start, gen_end, args):
    """Train generators [gen_start, gen_end) on the specified GPU.

    Each worker is a separate process. CUDA_VISIBLE_DEVICES must be set BEFORE
    importing torch, so we do all torch imports inside this function.
    """
    # Pin this process to a single physical GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import torch
    import torch.optim as optim
    from types import SimpleNamespace

    # Make sure we can find the parent FFF codebase
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(here, '..')))
    sys.path.insert(0, here)  # so we can import generative_canary

    from YOLOv8_Combiner import (
        add_defensivepatch_into_tensor, make_yolo_train_label,
        Yolov8Dataset, freeze_seed,
    )
    from ObjectDetector.fjn_yolov8 import FJN_YOLOv8 as YOLOv8
    from generative_canary import GenerativeCanary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    tag = f"[GPU {gpu_id}]"
    print(f"{tag} Starting worker for generators [{gen_start}, {gen_end}) on {device}",
          flush=True)

    # ---- Load detector ONCE ----
    print(f"{tag} Loading YOLOv8 detector...", flush=True)
    detector = YOLOv8()
    for p in detector.model.parameters():
        p.requires_grad = False

    cfg = SimpleNamespace(
        canary_cls_id=args.canary_cls_id, canary_size=80,
        img_size=args.img_size, person_conf=0.05, overlap_thresh=0.4,
        defensive_patch_location='cc', eval_no_overlap=True,
        margin_size=0, faster=False,
    )

    # ---- In-memory dataset (subclass that caches everything) ----
    class InMemoryYolov8Dataset(Yolov8Dataset):
        def __init__(self, *dargs, **dkwargs):
            super().__init__(*dargs, **dkwargs)
            # Use explicit parent class reference because super() doesn't
            # work inside list comprehensions (list comps have their own
            # scope and lose the __class__ cell).
            self._cache = [
                Yolov8Dataset.__getitem__(self, i)
                for i in range(self.img_ls_len)
            ]

        def __getitem__(self, idx):
            return self._cache[idx]

    print(f"{tag} Preloading training data into RAM...", flush=True)
    train_dataset = InMemoryYolov8Dataset(
        args.benign_root, args.adversarial_root, args.benign_label_root, args.img_size
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    print(f"{tag} Loaded {len(train_dataset)} image pairs", flush=True)

    # ---- Train each generator ----
    worker_start_time = time.time()
    completed = 0
    skipped = 0

    for gen_idx in range(gen_start, gen_end):
        ckpt_path = os.path.join(args.checkpoint_dir, f'gen_{gen_idx:04d}.pt')

        if os.path.exists(ckpt_path):
            skipped += 1
            if skipped == 1 or skipped % 50 == 0:
                print(f"{tag} Gen {gen_idx} already exists, skipping.", flush=True)
            continue

        gen_start_time = time.time()
        seed = gen_idx + 1
        freeze_seed(seed)

        g = GenerativeCanary(z_dim=args.z_dim).to(device)
        optimizer = optim.Adam(g.parameters(), lr=args.lr, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for _ in range(args.epochs):
            g.train()
            for benign_input, _, adv_input in train_loader:
                benign_input = benign_input.to(device)
                adv_input = adv_input.to(device)
                bs = benign_input.shape[0]

                # Per-image z and random bbox
                eta = torch.randn(bs, args.z_dim, device=device)
                bbox = torch.rand(bs, 4, device=device)
                z = torch.cat([eta, bbox], dim=1)

                canary_patches = g(z)  # (bs, 3, 80, 80)

                # Paste per-image
                adv_pos_all, ben_pos_all = [], []
                detector.model.eval()
                for bi in range(bs):
                    c_bi = canary_patches[bi]
                    _, ap = add_defensivepatch_into_tensor(
                        detector, cfg, adv_input[bi:bi + 1], c_bi, random_palce=True
                    )
                    _, bp = add_defensivepatch_into_tensor(
                        detector, cfg, benign_input[bi:bi + 1], c_bi, random_palce=True
                    )
                    adv_pos_all.extend(ap)
                    ben_pos_all.extend(bp)

                adv_batch = make_yolo_train_label(adv_input, adv_pos_all)
                ben_batch = make_yolo_train_label(benign_input, ben_pos_all)

                detector.model.train()
                adv_loss, _ = detector.model.model(adv_batch)
                ben_loss, _ = detector.model.model(ben_batch)

                loss = args.weight * ben_loss - adv_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

        # Save final checkpoint only (small format)
        torch.save({
            'generator_state_dict': g.state_dict(),
            'seed': seed,
            'gen_idx': gen_idx,
            'epoch': args.epochs,
        }, ckpt_path)

        elapsed = time.time() - gen_start_time
        completed += 1
        total_done = completed + skipped
        total_needed = gen_end - gen_start
        print(f"{tag} Generator {gen_idx}/{args.n_generators - 1} done "
              f"(seed={seed}) — {elapsed:.1f}s "
              f"[{total_done}/{total_needed} in this worker]",
              flush=True)

        # Periodic progress summary
        if completed % 50 == 0:
            wall = time.time() - worker_start_time
            rate = completed / wall if wall > 0 else 0
            remaining = (total_needed - total_done) / rate if rate > 0 else 0
            print(f"{tag} === Progress: {completed} trained, "
                  f"{skipped} skipped, "
                  f"{rate*3600:.1f} gen/hr, ETA {remaining/3600:.1f}h ===",
                  flush=True)

    total_wall = time.time() - worker_start_time
    print(f"{tag} DONE: {completed} trained, {skipped} skipped in {total_wall/60:.1f} min",
          flush=True)


def sequential_train(args):
    """Fallback when only 1 GPU is available — run in the main process."""
    print("Single GPU detected — running sequential training in main process.", flush=True)
    train_worker(0, 0, args.n_generators, args)


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Detect GPUs
    import torch
    import torch.multiprocessing as mp

    n_gpus_available = torch.cuda.device_count()
    if args.n_gpus == 0:
        n_gpus = n_gpus_available
    else:
        n_gpus = min(args.n_gpus, n_gpus_available)

    print(f"Detected {n_gpus_available} GPUs, using {n_gpus}")
    print(f"Training {args.n_generators} generators, {args.epochs} epochs each")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    if n_gpus <= 1:
        sequential_train(args)
        return

    # Warm up ultralytics cache on the main process so workers don't race
    # over the first download of yolov8n.pt
    print("Warming up ultralytics detector cache...")
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.abspath(os.path.join(here, '..')))
        from ObjectDetector.fjn_yolov8 import FJN_YOLOv8 as YOLOv8  # noqa
        _ = YOLOv8()
        del _
    except Exception as e:
        print(f"  Warmup failed (non-fatal): {e}")

    # Split generators across GPUs
    gens_per_gpu = args.n_generators // n_gpus
    remainder = args.n_generators % n_gpus

    chunks = []
    start = 0
    for gpu_id in range(n_gpus):
        extra = 1 if gpu_id < remainder else 0
        end = start + gens_per_gpu + extra
        chunks.append((gpu_id, start, end))
        start = end

    for gpu_id, s, e in chunks:
        print(f"  Worker GPU {gpu_id}: generators [{s}, {e}) ({e - s} gens)")

    mp.set_start_method('spawn', force=True)

    wall_start = time.time()
    processes = []
    for gpu_id, s, e in chunks:
        p = mp.Process(target=train_worker, args=(gpu_id, s, e, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_wall = time.time() - wall_start
    print(f"\nAll workers done. Total wall-clock time: {total_wall/3600:.2f}h "
          f"({total_wall/60:.1f} min)")

    # Count final checkpoints
    import glob
    ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'gen_*.pt')))
    print(f"Total checkpoints on disk: {len(ckpts)}")


if __name__ == '__main__':
    main()
