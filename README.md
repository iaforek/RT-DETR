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

## Compressed-byte sprite experiment

An independent synthetic experiment compares a pixel CNN against networks trained
directly on complete PNG/GIF bytes. It generates 64×64 indexed-colour sprites,
object masks, recolouring diffs and bounding-box labels, with separate train,
validation and test scenes. See [the experiment guide](experiments/compressed_sprites/README.md)
for generation, training, controls and local test commands. This does not change
the RT-DETR pipeline.

## Official basis

- Paper: *DETRs Beat YOLOs on Real-time Object Detection*, CVPR 2024
- Code: `https://github.com/lyuwenyu/RT-DETR`
- R18 configuration:
  `rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml`

## Selectable SpatialUNet encoder backbone

`--backbone presnet18` is the default and preserves existing experiments.
`--backbone spatial_unet` selects `SpatialUNetBackbone` from
`spatial_unet_model.py`. Both start from random initialisation unless a
compatible RT-DETR checkpoint is explicitly loaded.

This adaptation retains the supplied SpatialUNet encoder's double 3x3
convolutions, GroupNorm(8), GELU and widths 16/32/64/128/256. It adds a
512-channel stage at stride 32. The reconstruction decoder, sigmoid head
and pooled latent projection are not part of the detection backbone.
The RT-DETR hybrid encoder and detection decoder remain in use.

| Output | Source | Channels | Stride |
|---|---|---:|---:|
| S2 (optional) | encoder3 | 64 | 4 |
| S3 | encoder4 | 128 | 8 |
| S4 | bottleneck | 256 | 16 |
| S5 | added encoder5 | 512 | 32 |

The detector continues to load RGB images; the first convolution accepts
three channels. This patch does not import the grayscale OCT-F checkpoint
or change OCT-F training code. Input height and width must be multiples of
32, as already required by the training CLI. Base width is fixed at 16 in
the RT-DETR integration to keep the CLI and checkpoint contract small.

Examples (add your usual --root, --epochs and other training arguments):

```bash
python train.py --backbone spatial_unet --output-dir runs/spatial_unet
python train.py --backbone spatial_unet --use-p2 --output-dir runs/spatial_unet_p2
python train.py --backbone spatial_unet --use-p2 --use-spd --output-dir runs/spatial_unet_p2_spd
python train.py --backbone spatial_unet --use-s2-fusion --output-dir runs/spatial_unet_s2
```

`--use-spd` also works without P2. It replaces all five backbone max-pool
transitions with the existing SPDConv (space-to-depth, stride-1 convolution,
BatchNorm and GELU). The double-convolution blocks retain GroupNorm. The
existing SPD option also modifies bottom-up PAN downsampling. This makes
SPD a combined backbone/PAN ablation, not a backbone-only change.

P2 exposes S2 to a four-level encoder/decoder. S2 fusion instead injects S2
into P3 and keeps a three-level decoder. As before, S2 fusion is mutually
exclusive with P2 and global SPD; its own S2 fusion branch uses SPDConv.
`--use-spd-detr` remains the standalone FasterNet reconstruction and cannot
be combined with `--backbone spatial_unet`.

The backbone choice is recorded in config.json and checkpoint config.
Validation selects it automatically:

```bash
python validate.py --checkpoint runs/spatial_unet_p2/best.pt --root /path/to/data
```

For training resume, supply the same backbone and feature flags as the
original run, alongside `--resume runs/spatial_unet_p2/last.pt`.
Old checkpoints without a backbone field default to presnet18. Backbone
mismatches are rejected for configured pretrained/resume checkpoints.
`--pretrained` retains the existing same-layout RT-DETR weight loading
semantics; it is not an OCT-F autoencoder weight importer.
