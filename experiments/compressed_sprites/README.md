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

To run just one model, append `--modes png_bytes` (or `pixels`, `gif_bytes`).
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
prove that no byte-based architecture can work. Parameters/compute are not matched.

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
