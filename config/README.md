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

**Core scaling law experiments**

Approx [0.1, 0.2, 0.5, 1.0] x [0.5, 1, 2, 4] x 8h ~ 5 days

Split: 
- Base: 100%
- `p10`, `p20`, `p50`: 10%, 20%, 50% of training data

Model size: see above.

**Fine tuning**

Approx [0.1, 0.2, 0.5, 1.0] x 6 x 8h = 4 days

Objectives:
- Base: 3D map objective
- `bev`: Radarhd-style BEV
- `segment`: Semantic segmentation
- `vel`: Velocity estimation

Fine tune:
- `bev`: Radarhd-style BEV
- `segment`: Semantic segmentation
- `vel`: Velocity estimation


**Other ablations**

Approx 12 x 8h = 4 days

Individual vs separate: +3
- `bike`
- `indoor`
- `outdoor`

Patch: +2
- `dear`: balanced `patch(8, 1, 1, 16) -> (8, 8, 2, 16)`
- `rae`: range-azimuth-elevation patches `patch(64, 1, 1, 2) -> (1, 8, 2, 128)`
- `rd`: range-doppler patches `patch(2, 8, 2, 4) -> (32, 1, 1, 64)`
- `dea`: doppler-elevation-azimuth patches `patch(1, 1, 1, 128) -> (64, 8, 2, 2)`

Augmentation: +2
- Base: scalar + crop augmentation
- `aug.lite`: only scalar augmentation
- `aug.none`: no augmentation

Input representation: +5
- `base`: (doppler, amplitude + phase)
- `pad`: padded FFT
- `shuffle`: (shuffled slow time, amplitude + phase)
- `nofft`: (slow time, amplitude + phase)
- `amplitude`: (doppler, amplitude)
- `cfar`: (doppler, amplitude x cfar)
- `aoa`: (doppler, amplitude x cfar, aoa replacing azimuth)
