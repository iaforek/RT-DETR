# Detection directly from compressed sprite bytes

An isolated research experiment; the RT-DETR model and training pipeline are unchanged.
It tests whether a small network can locate and classify a single synthetic object
from complete PNG/GIF file bytes, without decoding those bytes into pixels.
This is a feasibility baseline, not evidence that compressed-byte detection is
inherently faster or more accurate, and not a reproduction of a published method.

## Run locally

Use the existing environment with PyTorch, NumPy and Pillow:

```powershell
conda activate diffraction-env
cd C:\Users\40464858\Development\RT-DETR
python experiments/compressed_sprites/experiment.py generate
python experiments/compressed_sprites/experiment.py train --device cuda
```

Use `--device cpu` without CUDA. Default generation: 3,000 training, 500 validation,
500 test scenes; default training: 30 epochs per model. Existing dataset/output
directories are refused to prevent accidental overwriting. For additional seeds:

```powershell
python experiments/compressed_sprites/experiment.py train --seed 43 --device cuda --output experiments/compressed_sprites/runs/seed43
```

To run just one model, append `--modes raw_bytes` (or `pixels`, `png_bytes`, `gif_bytes`).
The default now includes all four modes. Existing generated datasets already contain
`indices.bin`; no regeneration is needed for the new raw-index control.

For the three-way, 100-epoch comparison on Kelvin2, run from the repository root
in your existing PyTorch environment (inside your usual allocated GPU job):

```bash
python experiments/compressed_sprites/experiment.py train \
  --device cuda --modes pixels raw_bytes png_bytes --epochs 100 \
  --output experiments/compressed_sprites/runs/seed42_raw_comparison
```

To train only the added model on the same dataset:

```bash
python experiments/compressed_sprites/experiment.py train \
  --device cuda --modes raw_bytes --epochs 100 \
  --output experiments/compressed_sprites/runs/seed42_raw_only
```

Both commands start fresh training and keep earlier results intact. The raw model
saves `raw_bytes.pt` and a `raw_bytes` entry in `results.json`, whose test metrics
are under `test.indices.bin`.
To make a small smoke dataset, use a new `--data` directory and
`--train 24 --val 8 --test 8`; train it with `--epochs 1` and a new `--output`.
A smoke run checks execution only; it does not establish learning or generalisation.

Run dataset integrity tests (lossless round-trips, palette reconstruction,
mask/diff/box agreement and split organization):

```powershell
python -m unittest discover -s experiments/compressed_sprites -p "test_*.py" -v
```

## Data and labels

Each independent 64x64 scene contains one rectangle, ellipse or triangle with
random position, dimensions (4–32 pixels), foreground/background colours, or no
object (15% probability). The task deliberately starts simple: no occlusion,
texture, multiple objects or antialiasing. Images use a shared 256-entry RGB palette.
The 8-bit index array is 4,096 bytes; the palette adds 768 bytes. This is indexed
colour, not three-channel 8-bit RGB, which would require 12,288 pixel bytes.

Each scene folder contains:

- `indices.bin`: 4,096 raw indices, row-major; `palette.rgb`: 256 RGB triples.
- `image.png`: lossless indexed PNG, compression level 6, no annotation metadata.
- `image_level0.png`, `image_level9.png`: identical pixels, different PNG compression.
- `image_reindexed.png`: identical RGB pixels with shuffled palette indices.
- `image.gif`: lossless palette GIF for the same scene.
- `recoloured.png`: only the annotated object's colour is changed.
- `diff.png`: binary difference of original/recoloured **decoded RGB** images.
- `mask.png`: binary ground-truth mask; generation asserts it equals the diff.
- `label.json`: class and normalized xyxy box (exclusive right/bottom); class zero
  indicates no object and has an ignored zero box.

The recoloured image and diff are visualization/annotation artifacts. The detector
never receives them as features: the object location must be predicted from the
original input alone. Compressing a recoloured image can alter many subsequent
bytes, including checksums: a compressed-file byte diff is not an object mask.
Boxes are derived from the exact object mask. All encoding variants of a scene
remain in its original split. Colours do not encode the object class.

## Models and controls

- `pixels`: conventional 2D CNN operating on decoded RGB float32 tensors.
- `raw_bytes`: 4,096 uncompressed palette indices from `indices.bin`, flattened
  row by row. Position `i` corresponds to `x = i % 64`, `y = i // 64`.
  Uses exactly the same embedding, 1D CNN, pooling and prediction head as PNG/GIF,
  with the same padding length, optimiser, losses and checkpoint selection.
  It adds no coordinate inputs and never reads masks or labels as features.
  The palette is fixed across the dataset, so it is not included in this input.
  Requires exactly 4,096 source bytes; malformed files are rejected.
- `png_bytes`: byte embeddings and a 1D CNN over the **entire PNG file**, including
  headers, palette and compressed payload. No PNG decoding is performed by this
  input path. Positional bins are retained before the prediction head.
- `gif_bytes`: the same byte architecture, trained separately on GIF files.

Each model predicts background/shape class and one spatial box. Byte streams are
padded to 8,192 tokens with a dedicated padding token, never silently truncated.
Excessively long files cause an explicit error; increase `--max-bytes` if needed.
Padding and embeddings can consume MORE memory than decoded pixels. A smaller
file is not automatically a smaller neural-network input or cheaper computation.
The architecture is intentionally a baseline; unsuccessful learning would not
prove that no byte-based architecture can work. Pixel versus byte models are not
parameter/compute matched. Raw, PNG and GIF share the same architecture (155,120
parameters), padded tensor shape and seed-reset initialization. Their amount of
padding and distribution of informative tokens still differ. GPU training is not
guaranteed deterministic merely by setting the seed.

The raw control tests whether this 1D network can learn image coordinates without
also interpreting compression. Strong raw localisation and weak PNG localisation
would implicate the compressed representation/format processing; weak performance
on both would motivate investigation of the sequence architecture and optimisation.
It does not isolate DEFLATE alone: PNG also contains headers, palette and filters.
Raw evaluation uses the original `indices.bin` only, not shuffled-palette variants.

Training uses class cross-entropy and positive-only L1/IoU box losses. Checkpoints
are selected only on validation data by correct-class recall at IoU >= .5 plus
0.01 times mean positive IoU as a continuous tie-break preference. This is not AP.
Final test data is evaluated only after checkpoint selection. The PNG model is
also tested on re-encoded and reindexed copies of the SAME held-out scenes;
these tests expose reliance on a particular serialization. The pixel model sees
the same decoded RGB values for all PNG variants and should give identical scores.
GIF cross-encoding generalisation is not tested; its model is trained separately.

`runs/seed42/results.json` records class accuracy, mean IoU on positive scenes,
correct-class recall at IoU .5, background false-positive rate, parameter count,
file size, input tensor size, model/pipeline timings, training time and CUDA peak
allocated training memory (null on CPU). Timings use warm-up and CUDA synchronization;
pipeline time includes loading and preprocessing, but is sensitive to OS disk caches
and hardware. They are simple measurements, not a rigorous deployment benchmark.
No CPU peak-memory estimate is fabricated. Checkpoints and epoch histories are saved.

Compare several seeds and enough epochs for learning before drawing conclusions.
Examine localization as well as class accuracy: file length alone can correlate
with shape/size. Success on clean sprites does not establish performance on real
particles; stronger follow-ups include clutter, colour/background shifts, new
encoders, multiple objects and matched-compute architectures. Inspect size-stratified
results before making claims about tiny objects.
