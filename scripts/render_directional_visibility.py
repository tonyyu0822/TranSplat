#!/usr/bin/env python3
"""Render three directional visibility functions as an RGB diagnostic image.

Bakes the per-Gaussian directional visibility function V_lm (see
radiance_transfer_TranSplat.py) for a trained model, then encodes visibility
from three orthogonal world directions (+X, +Y, +Z) into the R, G, B
channels of a rendered image. This is a diagnostic tool for inspecting the
self-shadowing bake independently of any particular relighting pair.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--dataset", type=Path, required=True,
                         help="Path to the dataset used to train --model")
    parser.add_argument("--model", type=Path, required=True,
                         help="Path to a trained model directory (contains cfg_args, point_cloud/)")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument(
        "--train-view",
        type=int,
        default=0,
        help="Zero-based training-camera index to render.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <model>/directional_visibility/train_view_<view>",
    )
    parser.add_argument("--num-samples", type=int, default=1200)
    parser.add_argument("--shadow-res", type=int, default=1024)
    parser.add_argument(
        "--contrast-floor",
        type=float,
        default=0.7,
        help="Enhanced display maps this visibility value to black and 1 to white.",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="surfel_splatting")
    parser.add_argument("--force-bake", action="store_true")
    parser.add_argument("--inside-conda", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def reexec_in_conda(args: argparse.Namespace) -> None:
    if args.inside_conda or os.environ.get("CONDA_DEFAULT_ENV") == args.conda_env:
        return
    conda = shutil.which("conda")
    if not conda:
        raise RuntimeError("conda is not available on PATH")
    command = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--inside-conda",
    ]
    os.execv(conda, command)


def load_cfg_args(model_path: Path) -> Namespace:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    return eval(cfg_path.read_text(), {"Namespace": Namespace})


def dataset_args(args: argparse.Namespace) -> Namespace:
    cfg = load_cfg_args(args.model)
    return Namespace(
        sh_degree=getattr(cfg, "sh_degree", 3),
        source_path=str(args.dataset),
        model_path=str(args.model),
        images=getattr(cfg, "images", "images"),
        resolution=getattr(cfg, "resolution", -1),
        white_background=getattr(cfg, "white_background", False),
        data_device=getattr(cfg, "data_device", "cuda"),
        eval=True,
        render_items=getattr(
            cfg,
            "render_items",
            ["RGB", "Alpha", "Normal", "Depth", "Edge", "Curvature"],
        ),
    )


def pipeline_args() -> Namespace:
    return Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=0.0,
        debug=False,
    )


def save_rgb(path: Path, image) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = image.detach().permute(1, 2, 0).cpu().numpy()
    bgr = np.clip(rgb[..., ::-1] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Could not save {path}")


def save_rgba(path: Path, image, alpha) -> None:
    import cv2
    import numpy as np

    rgb = image.detach().permute(1, 2, 0).cpu().numpy()
    alpha_np = alpha.detach().squeeze(0).cpu().numpy()
    # The rasterizer returns color premultiplied over black. Convert to straight
    # alpha so compositing the PNG over arbitrary backgrounds has clean edges.
    straight_rgb = np.zeros_like(rgb)
    visible = alpha_np > 1e-6
    straight_rgb[visible] = rgb[visible] / alpha_np[visible, None]
    straight_rgb = np.clip(straight_rgb, 0.0, 1.0)
    rgba = np.concatenate([straight_rgb, alpha_np[..., None]], axis=2)
    bgra = np.clip(rgba[..., [2, 1, 0, 3]] * 255.0 + 0.5, 0, 255).astype(
        np.uint8
    )
    if not cv2.imwrite(str(path), bgra):
        raise RuntimeError(f"Could not save {path}")


def save_legend(path: Path, contrast_floor: float) -> None:
    import cv2
    import numpy as np

    canvas = np.full((250, 900, 3), 24, dtype=np.uint8)
    entries = (
        ("R", "visibility from world +X", (40, 40, 230)),
        ("G", "visibility from world +Y", (40, 210, 40)),
        ("B", "visibility from world +Z", (230, 70, 40)),
    )
    for row, (channel, text, color) in enumerate(entries):
        y = 42 + row * 58
        cv2.rectangle(canvas, (28, y - 25), (76, y + 15), color, -1)
        cv2.putText(
            canvas,
            f"{channel}: {text}",
            (96, y + 7),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"Enhanced mapping: visibility {contrast_floor:.2f} -> black, 1.00 -> white",
        (28, 225),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not save {path}")


def main() -> int:
    args = parse_args()
    reexec_in_conda(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args.repo = args.repo.resolve()
    args.dataset = args.dataset.resolve()
    args.model = args.model.resolve()
    if args.output_dir is None:
        args.output_dir = args.model / "directional_visibility" / f"train_view_{args.train_view:03d}"
    args.output_dir = args.output_dir.resolve()
    input_ply = (
        args.model
        / "point_cloud"
        / f"iteration_{args.iteration}"
        / "point_cloud.ply"
    )
    for path in (args.dataset, args.model, input_ply):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0.0 <= args.contrast_floor < 1.0:
        raise ValueError("--contrast-floor must be in [0, 1)")

    sys.path.insert(0, str(args.repo))
    import torch

    from gaussian_renderer import render
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    import radiance_transfer_TranSplat as transfer

    model = GaussianModel(load_cfg_args(args.model).sh_degree)
    scene = Scene(
        dataset_args(args),
        model,
        load_iteration=args.iteration,
        shuffle=False,
    )
    train_cameras = scene.getTrainCameras()
    if not 0 <= args.train_view < len(train_cameras):
        raise IndexError(
            f"Training view {args.train_view} is outside [0, {len(train_cameras) - 1}]"
        )
    camera = train_cameras[args.train_view]

    gaussians = {
        "xyz": model._xyz,
        "features_dc": model._features_dc,
        "features_rest": model._features_rest,
        "opacity": model._opacity,
        "scaling": model._scaling,
        "rotation": model._rotation,
        "normal": None,
        "max_sh_degree": model.max_sh_degree,
    }
    device = "cuda"
    (
        directions,
        sqrt_weights,
        sh_basis,
        basis_weighted,
        ata_weighted,
        _,
        _,
    ) = transfer.precompute_sh_sampling(
        args.num_samples, model.max_sh_degree, device
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        args.output_dir
        / f"vlm_cache_samples{args.num_samples}_res{args.shadow_res}.pt"
    )
    if cache_path.is_file() and not args.force_bake:
        print(f"Loading visibility cache: {cache_path}", flush=True)
        visibility_lm = torch.load(cache_path, map_location=device)["V_lm"].to(device)
        if visibility_lm.shape[0] != model._xyz.shape[0]:
            raise ValueError("Visibility cache does not match the loaded checkpoint")
    else:
        print("Baking directional visibility V_lm...", flush=True)
        with torch.no_grad():
            visibility_lm = transfer.phase_1_bake_visibility_sh(
                gaussians,
                directions,
                basis_weighted,
                ata_weighted,
                sqrt_weights,
                device,
                res=args.shadow_res,
            )
        torch.save({"V_lm": visibility_lm.cpu()}, cache_path)

    # Physical world directions encoded in RGB. The visibility bake stores its
    # SH function with X flipped, so map world directions into that convention.
    world_directions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    sh_directions = world_directions.clone()
    sh_directions[:, 0] *= -1.0
    directional_bases = transfer.evaluate_sh_bases_torch(
        sh_directions, model.max_sh_degree, device
    )
    visibility_rgb = (visibility_lm @ directional_bases.T).clamp(0.0, 1.0)
    visibility_enhanced = (
        (visibility_rgb - args.contrast_floor) / (1.0 - args.contrast_floor)
    ).clamp(0.0, 1.0)

    background = torch.zeros(3, dtype=torch.float32, device=device)
    pipe = pipeline_args()
    outputs = {
        "directional_visibility_rgb_raw.png": visibility_rgb,
        "directional_visibility_rgb_enhanced.png": visibility_enhanced,
    }
    channel_names = ("r_pos_x", "g_pos_y", "b_pos_z")
    for channel, name in enumerate(channel_names):
        scalar = visibility_rgb[:, channel : channel + 1].expand(-1, 3)
        outputs[f"visibility_{name}.png"] = scalar

    with torch.no_grad():
        for filename, colors in outputs.items():
            package = render(
                camera,
                model,
                pipe,
                background,
                override_color=colors,
            )
            save_rgb(args.output_dir / filename, package["render"])
        grayscale = visibility_rgb.mean(dim=1, keepdim=True).expand(-1, 3)
        package = render(
            camera,
            model,
            pipe,
            background,
            override_color=grayscale,
        )
        grayscale_rgba_name = "directional_visibility_grayscale_rgba.png"
        save_rgba(
            args.output_dir / grayscale_rgba_name,
            package["render"],
            package["rend_alpha"],
        )
    save_legend(args.output_dir / "legend.png", args.contrast_floor)

    quantile_levels = torch.tensor(
        [0.01, 0.5, 0.99], dtype=torch.float32, device=device
    )
    quantiles = torch.quantile(visibility_rgb, quantile_levels, dim=0).cpu()
    manifest = {
        "object": args.model.name,
        "camera_split": "train",
        "camera_index_zero_based": args.train_view,
        "human_ordinal_view": args.train_view + 1,
        "checkpoint": str(input_ply),
        "visibility_method": "ESM+PCF depth-map bake projected to SH V_lm",
        "rgb_encoding": {
            "R": "visibility from world +X",
            "G": "visibility from world +Y",
            "B": "visibility from world +Z",
        },
        "enhanced_mapping": (
            f"clamp((visibility - {args.contrast_floor}) / "
            f"{1.0 - args.contrast_floor}, 0, 1)"
        ),
        "num_samples": args.num_samples,
        "shadow_resolution": args.shadow_res,
        "per_channel_visibility_quantiles_p01_p50_p99": {
            channel_names[channel]: quantiles[:, channel].tolist()
            for channel in range(3)
        },
        "outputs": [*outputs, grayscale_rgba_name, "legend.png"],
        "cache": str(cache_path),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Saved RGB directional visibility: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
