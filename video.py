"""Render comparison video."""

import argparse
import os

import numpy as np
import yaml
from beartype.typing import cast
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import Array, UInt
from roverd import Dataset, sensors
from tqdm import tqdm

from roverp import CFARProcessing, doppler_range_azimuth_elevation, graphics


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path (trace name).")
    p.add_argument(
        "-o", "--out", default=None, help="Output path.")
    p.add_argument("-d", "--data", default="data", help="Path to datasets.")
    p.add_argument(
        "-r", "--results", default="results", help="Path to results.")
    p.add_argument(
        "--font", default=None, help="Use a specific `.ttf` font file.")
    p.add_argument(
        "-f", "--fps", type=float, default=30.0,
        help="Output video framerate.")
    p.add_argument(
        "-s", "--timescale", type=float, default=1.0,
        help="Real time to video time scale factor (larger = faster)")
    p.add_argument(
        "--anon", default=False, action='store_true',
        help="Anonymize trace name (keeping root, but hiding place name).")
    p.add_argument(
        "--duration", type=float,
        help="Stop early after `--duration` seconds.")


def _load(args):
    """Load timestamps and streams."""
    ds = Dataset(os.path.join(args.data, args.path))

    models = {
        "base": ("small/base", "depth"),
        "seg": ("seg/f100", "seg"),
        "bev": ("bev/f100", "bev")
    }

    align = np.load(os.path.join(
        args.data, args.path, "_fusion", "indices.npz"))
    ts_model = ds[align["sensors"][0]].timestamps()[align["indices"][:, 0]]

    streams = {
        "lidar": cast(
            sensors.LidarData, ds["lidar"]).destaggered_stream("rng"),
        "camera": ds["camera"]["video.avi"].stream_prefetch(),
        "radar": cast(
            sensors.RadarData, ds["radar"]).iq_stream(prefetch=True, batch=0),
    }
    timestamps = {
        "radar": ds["radar"].timestamps(),
        "camera": ds["camera"].timestamps(),
        "lidar": ds["lidar"].timestamps(),
    }

    for k, (path, key) in models.items():
        streams[k] = iter(sensors.SensorData(
            os.path.join(args.results, path, "eval", args.path))[key].read())
        timestamps[k] = ts_model
    streams["bev_gt"] = iter(sensors.SensorData(
        os.path.join(args.results, "bev/f100", "eval", args.path)
    )["bev_gt"].read())
    streams["seg_gt"] = iter(sensors.SensorData(
        os.path.join(args.results, "seg/f100", "eval", args.path)
    )["seg_gt"].read())
    timestamps["bev_gt"] = ts_model
    timestamps["seg_gt"] = ts_model

    return timestamps, streams


def _renderer(args):
    """Create (and close on) renderer."""
    viridis = graphics.mpl_colormap('viridis')
    inferno = graphics.mpl_colormap('inferno')

    with open("schema/colors.yaml") as f:
        colors = np.array(yaml.load(f, Loader=yaml.SafeLoader)["colors"])

    def radar_tf(iq):
        # doppler-tx-rx-range
        drae = doppler_range_azimuth_elevation(iq)[:, ::-1, :, :]
        concat = rearrange(drae, "d r a e -> (e r) (a d)")
        return graphics.render_image(
            concat, viridis, resize=(640, 2560), pmin=1.0, pmax=99.9)

    def lidar_tf(x):
        return graphics.render_image(
            x[:, 512:-512], colors=viridis, resize=(640, 1280),
            scale=11000 if args.path.startswith("indoor") else 22000)

    def camera_tf(img):
        crop = img[60:-60]
        return graphics.resize(crop, height=640, width=1280)

    def depth_tf(y: UInt[Array, "h w"]):
        return graphics.render_image(
            y, colors=viridis, resize=(640, 1280), scale=64.0)

    def bev_tf(y):
        y = y[::-1, ::-1]
        return graphics.render_image(y, colors=inferno, resize=(640, 1280))

    def seg_tf(y):
        return graphics.resize(
            jnp.take(colors, y, axis=0), height=640, width=1280)

    ds = Dataset(os.path.join(args.data, args.path))
    sensor = cast(sensors.RadarData, ds["radar"])
    sample = sensor.iiqq16_to_iq64(sensor['iq'].read(samples=1024))
    proc_cfar = CFARProcessing(
        jnp.array(sample), max_points=256, cfar_threshold=0.001)

    def cfar_tf2d(iq):
        rng, az, _, intensity = proc_cfar(iq[None, ...])
        x = jnp.sin(az) * rng
        y = jnp.cos(az) * rng

        left, right = jnp.percentile(intensity, jnp.array([5, 98]))
        intensity = (jnp.clip(intensity, left, right) - left) / (right - left)
        intensity = intensity**0.5

        scattered = graphics.Scatter(
            radius=6.2, resolution=(640, 1280)
        )(x[0] / 512 + 0.5, y[0] / 256, intensity[0])
        return graphics.render_image(scattered, colors=inferno)

    def cfar_tf3d(iq):
        rng, az, el, _ = proc_cfar(iq[None, ...])
        scattered = graphics.Scatter(
            radius=6.2, resolution=(640, 1280)
        )(az / (np.pi / 2), el / (np.pi / 4), rng / 256)
        return graphics.render_image(scattered, colors=viridis)


    return graphics.Render(
        size = (2560, 3840),
        channels={
            (0, 640, 1280, 3840): "radar",
            (640, 1280, 0, 1280): "cfar2",
            (1280, 1920, 0, 1280): "bev",
            (1920, 2560, 0, 1280): "bev_gt",
            (640, 1280, 1280, 2560): "cfar3",
            (1280, 1920, 1280, 2560): "base",
            (1920, 2560, 1280, 2560): "lidar",
            (640, 1280, 2560, 3840): "camera",
            (1280, 1920, 2560, 3840): "seg",
            (1920, 2560, 2560, 3840): "seg_gt"
        },
        transforms={
            "lidar": lidar_tf,
            "camera": camera_tf,
            "radar": radar_tf,
            "base": depth_tf,
            "seg": seg_tf,
            "seg_gt": seg_tf,
            "bev": bev_tf,
            "bev_gt": bev_tf,
            "cfar2": cfar_tf2d,
            "cfar3": cfar_tf3d
        },
        text={
            (20, 40): "GRT Small IQ-1M",
            (100, 40): (
                (args.path.split("/")[0] + "/<anonymous.location>")
                if args.anon else args.path),
            (180, 40): "+{mm:02d}:{ss:05.2f}s",
            (260, 40): "radar   {radar:06d}",
            (340, 40): "lidar   {lidar:06d}",
            (420, 40): "camera  {camera:06d}",
        #     (20, 1280 + 40): "4D Radar Data Cube",
        #     (640 + 540, 40): "Radar Depth",
        #     (1280 + 540, 40): "Lidar Depth",
        #     (640 + 540, 1280 + 40): "Radar Segmentation",
        #     (1280 + 540, 1280 + 40): "Camera Reference",
        #     (640 + 540, 2560 + 40): "Radar BEV",
        #     (1280 + 540, 2560 + 40): "Lidar BEV"
        },
        font=graphics.JaxFont(args.font, size=60),
        textcolor=(255, 255, 255)
    ).render


def _main(args):

    if args.out is None:
        args.out = "compare.mp4"

    timestamps, streams = _load(args)
    render_func = _renderer(args)
    frame_time = 1 / args.fps * args.timescale

    est_n_frames = int((
        min(v[-1] for v in timestamps.values())
        - max(v[0] for v in timestamps.values())
    ) / frame_time) // 32

    synced = graphics.synchronize(
        streams, timestamps, period=frame_time, round=1.0,
        duplicate={"cfar2": "radar", "cfar3": "radar"}, batch=32,
        stop_at=0.0 if args.duration is None else args.duration)

    def _stack(batch):
        stacks = {
            k: jnp.stack([frameset[k] for _, _, frameset in batch])
            for k in batch[0][2]}
        captions = [
            {"mm": int(t / 60), "ss": (t % 60), **ii} for t, ii, _ in batch]
        return stacks, captions

    render_iter = (
        render_func(*_stack(batch))
        for batch in tqdm(synced, total=est_n_frames))

    graphics.write_consume(
        render_iter, out=args.out, fps=args.fps, codec="h264", queue_size=4)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    _parse(p)
    _main(p.parse_args())
