# RT-DETR-R18 From Scratch

Compact RT-DETR-R18 recreation for direct comparison with the
YOLOv3 and YOLOv26 learning repositories.

## Files

- `model.py` — PResNet-18, HybridEncoder, top-k query selection, deformable
  transformer decoder, denoising queries and NMS-free prediction.
- `train.py` — YOLO-format VOC dataset, Hungarian matching, Varifocal Loss,
  L1 box loss, GIoU loss, training/validation loops and checkpoints.
- `validate.py` — full-dataset AP/mAP validation or single-image annotation.
- `checkpoint_selection.py` — validation AP selection policy and resume checks.

The architecture follows the original RT-DETR PyTorch implementation and its
R18 configuration: PResNet-18 variant d, 256-dimensional hybrid encoder, 300
queries, three decoder layers and 100 denoising queries. The code is trained
from random initialisation and does not load pretrained weights.

This compact repository is intended for architecture study and controlled
experiments. It is not checkpoint-compatible with the multi-file official
RT-DETR repository.

## Dependencies

```bash
pip install numpy pillow scipy
```

Use an environment with PyTorch installed. GPU training requires a CUDA-enabled
PyTorch installation; the local regression tests run on CPU. The local
`diffraction-env` environment has successfully run all eight tests.

## Dataset layout

```text
VOC_dataset/voc_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Labels use normalized YOLO format:

```text
class_id x_center y_center width height
```

## Train

```bash
python train.py \
  --root /mnt/scratch2/users/40464858/VOC_dataset/voc_yolo \
  --img-size 416 \
  --batch-size 16 \
  --epochs 100 \
  --output-dir runs/rtdetr_voc
```

Resume:

```bash
python train.py \
  --resume runs/rtdetr_voc/last.pt \
  --root /mnt/scratch2/users/40464858/VOC_dataset/voc_yolo \
  --output-dir runs/rtdetr_voc \
  --epochs 100
```

Checkpoints and history are written to:

```text
runs/rtdetr_voc/
├── best.pt
├── last.pt
├── config.json
└── history.jsonl
```

`best.pt` is selected by **highest validation mAP@0.50:0.95**, measured after
every epoch. Equal scores retain the earlier checkpoint. `last.pt` always stores
the latest completed epoch for resuming. Validation loss remains a diagnostic;
it does not select `best.pt`.

AP uses the same inference and matching code as `validate.py`, with confidence
cutoff 0.001 and at most 300 detections per image. This is the repository's custom
AP protocol, not official COCOeval. The extra inference/AP pass increases epoch
time. Use a fixed validation split for selection and a separate held-out test
split for final reporting.

History includes validation `map50`, `map50_95`, `best_validation_ap`, and
`best_ap_epoch`. Checkpoints store the same selection state and evaluation
protocol. Epoch fields in checkpoint files are zero-based; history epochs are
one-based. Resume with the same dataset, architecture, image size and AMP setting.

Older loss-selected checkpoints remain resumable. Their resumed weights are
evaluated to establish an AP baseline before further training. Any existing
`best.pt` in the output directory is preserved as `best_before_ap_selection.pt`
(with a numbered suffix if needed). The AP baseline and subsequent epochs compete
for the new `best.pt`; this cannot recover the best AP of discarded historical
epochs, and the archived loss-best checkpoint is not automatically evaluated.
For a new experiment, use a new output directory.

## Run tests locally

The tests use Python's built-in `unittest`; **pytest is not required**. Activate
an environment containing PyTorch, NumPy, SciPy and Pillow. In a PowerShell
terminal configured for Conda, run:

```powershell
conda activate diffraction-env
cd C:\Users\40464858\Development\RT-DETR
python -m unittest discover -s .\tests -v
```

Replace the environment name and repository path if your installation differs.
Run discovery from the repository root so it finds this project's test files.

To run each file separately:

```powershell
python -m unittest discover -s .\tests -p "test_checkpoint_selection.py" -v
python -m unittest discover -s .\tests -p "test_training_ap.py" -v
```

The full suite currently contains eight tests. Successful output ends with:

```text
Ran 8 tests in ...

OK
```

`test_checkpoint_selection.py` checks invalid AP rejection, zero initial scores,
strict improvement, and restoration of the best AP and evaluation protocol.
`test_training_ap.py` checks AP-based selection despite lower loss at another
epoch, resume and tie handling, legacy checkpoint preservation, empty validation
labels, and agreement between logged AP and a reloaded real-model checkpoint.

The integration tests run on CPU and create temporary images and checkpoints,
which are cleaned up automatically. They do not require your research datasets
or modify existing experiment results. Passing them verifies these code paths;
it does not establish full-dataset accuracy or GPU stability.

### Troubleshooting

- If a traceback points to `site-packages\tests\test_cli.py` and complains about
  missing `pytest`, an unrelated installed test package was discovered. Use the
  repository-root discovery commands above rather than an import such as
  `python -m unittest tests.test_training_ap`. Installing pytest is not needed
  for this project's tests.
- Some PyTorch versions emit a `FutureWarning` from `torch.load()` because the
  resume loader does not explicitly specify `weights_only`. This warning does
  not fail the tests; the test checkpoints are generated within the tests.
  Check that the final result is `OK`. Outside the tests, only load checkpoints
  from trusted sources.

## Validate the complete VOC split

```bash
python validate.py \
  --checkpoint runs/rtdetr_voc/best.pt \
  --root /mnt/scratch2/users/40464858/VOC_dataset/voc_yolo \
  --split val \
  --batch-size 32
```

The script reports the same comparison metrics as the YOLO validators:

- Precision and recall, micro and macro
- F1, micro and macro
- TP, FP and FN
- mean matched IoU
- AP50 and AP50-95 per class
- mAP@0.50 and mAP@0.50:0.95
- inference time

It saves:

```text
runs/rtdetr_voc/validation/
├── metrics.json
└── per_class_metrics.csv
```

## Annotate one image

```bash
python validate.py \
  --checkpoint runs/rtdetr_voc/best.pt \
  --image /mnt/scratch2/users/40464858/coco128/images/train2017/000000000113.jpg \
  --conf-thres 0.05 \
  --out-path pred_vis_rtdetr.jpg
```

RT-DETR uses its native NMS-free decoder-query output. No NMS parameter is
required.

## Official basis

- Paper: *DETRs Beat YOLOs on Real-time Object Detection*, CVPR 2024
- Code: `https://github.com/lyuwenyu/RT-DETR`
- R18 configuration:
  `rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml`
