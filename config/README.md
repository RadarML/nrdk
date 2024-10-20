# Deep Radar Configuration

## Model Sizes

All models use a head size of 64 dims per head. Peak throughput measures
batched frames per second with a sweep over (1, 2, 4, ... 32) frames per batch.

| Size    | Layers | Dimension      | Params | Peak Throughput |
| ------- | ------ | -------------- | ------ | --------------- |
| Tiny    | 3 + 3  | 384 (6 heads)  | 12.7M  | 316.1 fps       |
| Small   | 4 + 4  | 512 (8 heads)  | 28.9M  | 168.6 fps       |
| Medium  | 6 + 6  | 640 (10 heads) | 69.4M  | 83.9 fps        |
| Large   | 9 + 9  | 768 (12 heads) | 149M   | 44.3 fps        |

## Ablations

**Core scaling law experiments** (16)

Split: training data percent

- Base: 100%
- `p10`, `p20`, `p50`: 10%, 20%, 50% of training data

Model size: see above.

**Other ablations**

Patch: patch size (+2)

- Base: balanced `patch(8, 1, 1, 6) -> (8, 8, 2, 16)`
- `rae`: `patch(64, 1, 1, 2) -> (1, 8, 2, 128)`
- `rd`: `patch(2, 8, 2, 4) -> (32, 1, 1, 64)`

Augmentation: (+2)
- Base: scalar + crop augmentation
- `aug.lite`: only scalar augmentation
- `aug.none`: no augmentation

Input representation: (+5)
- `base`: (doppler, amplitude + phase)
- `shuffle`: (shuffled slow time, amplitude + phase)
- `nofft`: (slow time, amplitude + phase)
- `amplitude`: (doppler, amplitude)
- `cfar`: (doppler, amplitude x cfar)
- `aoa`: (doppler, amplitude x cfar, aoa replacing azimuth)
