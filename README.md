# RT-DETR-R18 From Scratch

Compact three-file RT-DETR-R18 recreation for direct comparison with the
YOLOv3 and YOLOv26 learning repositories.

## Files

- `model.py` — PResNet-18, HybridEncoder, top-k query selection, deformable
  transformer decoder, denoising queries and NMS-free prediction.
- `train.py` — YOLO-format VOC dataset, Hungarian matching, Varifocal Loss,
  L1 box loss, GIoU loss, training/validation loops and checkpoints.
- `validate.py` — full-dataset AP/mAP validation or single-image annotation.

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

Use the existing CUDA-enabled PyTorch installation in the `yolo_cuda`
environment.

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
