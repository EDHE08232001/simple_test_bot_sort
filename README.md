# BoT-SORT lab

Two ways to explore the BoT-SORT tracker (Ultralytics YOLO + BoT-SORT), each comparing
**A0 = camera-motion compensation (GMC) off** with **A1 = GMC on**.

```
source venv/bin/activate
python run.py                 # interactive menu
```

## 1. Play with the camera

```
python run.py live                  # keys: g toggle GMC, r reset tracker, q quit
python run.py record clip.mp4 --seconds 20
```
macOS asks for camera permission for your terminal the first time.

## 2. Benchmark on labeled data

```
python run.py evaluate --dataset mot17  --data /path/to/MOT17
python run.py evaluate --dataset lasot  --data /path/to/LaSOT --max-seqs 5
```
Useful flags: `--seqs MOT17-02 MOT17-04`, `--max-frames 200` (quick test), `--arms gmc` (one arm only),
`--weights yolo26n.pt`, `--device cpu|mps|0`.

When the run finishes it prints the tables, then asks **"Visualize the metrics as diagrams? [y/N]"**
(`--plot` / `--no-plot` answer it up front). Diagrams are always saved as PNGs under `results/<run>/plots/`.
Redraw them later with `python run.py plot results/<run>`.

Each run writes `results/<dataset>_<timestamp>/` with `results.json`, `summary.csv`, and the tracker
output in MOTChallenge format (`tracks/<arm>/<sequence>.txt`).

### Expected dataset layout
- **MOT17**: `MOT17/train/MOT17-02-FRCNN/{img1/, gt/gt.txt, seqinfo.ini}`. The three detector variants share
  images and labels, so only one (`--mot-variant`, default FRCNN) is run. The `test` split has no labels.
- **LaSOT**: `LaSOT/<class>/<class>-<n>/{img/, groundtruth.txt, out_of_view.txt}`.

### Metrics
- **MOT17** (multi-object): HOTA, DetA, AssA, MOTA, MOTP, IDF1, ID switches, FP, FN, MT/ML, FPS.
  Implemented in `botsort_lab/metrics_mot.py` following the MOTChallenge / TrackEval definitions
  (pedestrians only; predictions on distractor regions are ignored).
- **LaSOT** (single-object): success AUC, precision@20px, normalized precision AUC, lost %, FPS.
  BoT-SORT is multi-object, so the track overlapping the ground-truth box (IoU >= 0.3) is acquired and then
  followed by ID. Losing it or an ID switch counts as failure.

FPS covers detection + tracking only (no image decoding or drawing).

## Layout
```
run.py                 entry point
botsort_lab/
  cli.py               commands / menu
  live.py              camera + recording
  evaluate.py          runs arms over sequences, saves results
  datasets.py          MOT17 + LaSOT loaders
  metrics_mot.py       CLEAR-MOT, IDF1, HOTA
  metrics_sot.py       success / precision
  report.py            tables, CSV, diagrams
  tracking.py, config.py
tracker_cfgs/          generated BoT-SORT yaml per arm
```
