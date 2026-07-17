#!/usr/bin/env python3
"""Create fixed-camera object demos under a rotating environment map.

Visibility is baked once. For every yaw, the script rotates the target HDR
using a horizontal-roll convention, relights in memory via
radiance_transfer_TranSplat, and directly renders the selected test cameras.
This avoids writing hundreds of redundant intermediate PLY files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Render fixed object views while the target HDR rotates."
    )
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--dataset", type=Path, required=True,
                         help="Path to the dataset used to train --model")
    parser.add_argument("--model", type=Path, required=True,
                         help="Path to a trained model directory (contains cfg_args, point_cloud/)")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument(
        "--object-name",
        default="",
        help="Display/output name. Defaults to the model directory name.",
    )
    parser.add_argument("--source-env", type=Path, required=True,
                         help="Source HDR env map the model was implicitly lit by")
    parser.add_argument("--target-env", type=Path, required=True,
                         help="Target HDR env map to rotate around the object")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <model>/rotating_light_demo/<target-env stem>",
    )
    parser.add_argument("--view-indices", nargs="+", type=int, default=[0],
                         help="Zero-based test-camera indices to render (one video per index)")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1500)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--conda-env", default="surfel_splatting")
    parser.add_argument("--num-samples", type=int, default=1200)
    parser.add_argument("--shadow-res", type=int, default=1024)
    parser.add_argument("--floor-alpha", type=float, default=0.05)
    parser.add_argument("--tau-max", type=float, default=3.0)
    parser.add_argument("--specular-boost", type=float, default=1.0)
    parser.add_argument(
        "--video-layout",
        default="comparison",
        choices=("comparison", "render-only"),
        help="Use the original panorama/render comparison or encode only the render.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-relight-render",
        action="store_true",
        help="Only rebuild videos from existing HDR previews and render frames.",
    )
    parser.add_argument("--skip-video", action="store_true")
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


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def degree_tag(degrees: float) -> str:
    text = f"{degrees:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = text.zfill(3)
    return text.replace("-", "m").replace(".", "p")


def yaw_degrees(frame_index: int, frame_count: int) -> float:
    return frame_index * 360.0 / frame_count


def hdr_path(args: argparse.Namespace, index: int) -> Path:
    yaw = yaw_degrees(index, args.frames)
    return (
        args.output_dir
        / "rotated_envmaps"
        / "hdr"
        / f"{args.target_name}_yaw{degree_tag(yaw)}.hdr"
    )


def preview_path(args: argparse.Namespace, index: int) -> Path:
    return args.output_dir / "rotated_envmaps" / "preview" / f"{index:05d}.png"


def render_path(args: argparse.Namespace, view_index: int, frame_index: int) -> Path:
    return (
        args.output_dir
        / "renders"
        / f"test_{view_index:03d}"
        / f"{frame_index:05d}.png"
    )


def validate(args: argparse.Namespace) -> Path:
    args.repo = args.repo.resolve()
    args.dataset = args.dataset.resolve()
    args.model = args.model.resolve()
    args.source_env = args.source_env.resolve()
    args.target_env = args.target_env.resolve()
    model_name = args.model.name.lower()
    if model_name.endswith("_vanilla"):
        model_name = model_name[: -len("_vanilla")]
    args.object_name = args.object_name.strip().lower() or model_name
    args.target_name = args.target_env.stem.lower()
    if args.output_dir is None:
        args.output_dir = args.model / "rotating_light_demo" / args.target_name
    args.output_dir = args.output_dir.resolve()

    input_ply = args.model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    for path in [
        args.repo / "radiance_transfer_TranSplat.py",
        args.source_env,
        args.target_env,
        input_ply,
    ]:
        require_file(path)
    if not args.dataset.is_dir() or not args.model.is_dir():
        raise FileNotFoundError("Dataset or model directory is missing")
    if args.frames <= 0 or args.fps <= 0:
        raise ValueError("--frames and --fps must be positive")
    if len(args.view_indices) != len(set(args.view_indices)):
        raise ValueError("--view-indices must not contain duplicates")
    if min(args.view_indices) < 0:
        raise ValueError("--view-indices must be non-negative")
    if args.width % 2 or args.height % 2:
        raise ValueError("Video dimensions must be even for yuv420p")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not available on PATH")
    return input_ply


def outputs_complete(args: argparse.Namespace) -> bool:
    for index in range(args.frames):
        if not hdr_path(args, index).is_file() or not preview_path(args, index).is_file():
            return False
        for view_index in args.view_indices:
            if not render_path(args, view_index, index).is_file():
                return False
    return True


def load_cfg_args(model_path: Path) -> Namespace:
    cfg_path = model_path / "cfg_args"
    require_file(cfg_path)
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


def rotate_envmap(env_bgr, yaw: float):
    import numpy as np

    shift = int(round((yaw / 360.0) * env_bgr.shape[1]))
    return np.roll(env_bgr, shift=shift, axis=1)


def linear_to_srgb(image):
    import numpy as np

    image = np.maximum(image, 0.0)
    image = image / (1.0 + image)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    )


def save_hdr_and_preview(args: argparse.Namespace, index: int, rotated_bgr) -> None:
    import cv2
    import numpy as np

    hdr_output = hdr_path(args, index)
    preview_output = preview_path(args, index)
    hdr_output.parent.mkdir(parents=True, exist_ok=True)
    preview_output.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not hdr_output.is_file():
        if not cv2.imwrite(str(hdr_output), rotated_bgr):
            raise RuntimeError(f"Could not save {hdr_output}")
    if args.force or not preview_output.is_file():
        preview = np.clip(
            linear_to_srgb(rotated_bgr.astype(np.float32)), 0.0, 1.0
        )
        preview = (preview * 255.0 + 0.5).astype(np.uint8)
        cv2.imwrite(str(preview_output), preview)


def composite_envmap(rgb, alpha, camera, envmap_tensor, bg_color):
    import torch

    height = camera.image_height
    width = camera.image_width
    fx = 0.5 * width / math.tan(0.5 * camera.FoVx)
    fy = 0.5 * height / math.tan(0.5 * camera.FoVy)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device="cuda"),
        torch.arange(width, dtype=torch.float32, device="cuda"),
        indexing="ij",
    )
    dirs_camera = torch.stack(
        [
            (x - width / 2.0 + 0.5) / fx,
            (y - height / 2.0 + 0.5) / fy,
            torch.ones_like(x),
        ],
        dim=-1,
    )
    dirs_camera /= torch.norm(dirs_camera, dim=-1, keepdim=True)
    dirs_world = torch.matmul(
        dirs_camera, camera.world_view_transform[:3, :3].T
    )
    xs, ys, zs = -dirs_world[..., 0], dirs_world[..., 1], dirs_world[..., 2]
    theta = torch.acos(zs.clamp(-1.0, 1.0))
    phi = torch.atan2(ys, xs)
    phi = torch.where(phi < 0, phi + 2 * math.pi, phi)
    grid = torch.stack(
        [(phi / (2 * math.pi)) * 2 - 1, (theta / math.pi) * 2 - 1],
        dim=-1,
    ).unsqueeze(0)
    env = envmap_tensor.permute(2, 0, 1).unsqueeze(0)
    background = torch.nn.functional.grid_sample(
        env,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0)
    background = torch.where(
        background <= 0.0031308,
        background * 12.92,
        1.055 * torch.pow(background.clamp(min=1e-10), 1.0 / 2.4) - 0.055,
    ).clamp(0.0, 1.0)
    bg_tensor = torch.tensor(
        bg_color, dtype=torch.float32, device="cuda"
    ).view(3, 1, 1)
    foreground = (rgb - bg_tensor * (1.0 - alpha)).clamp(0.0, 1.0)
    return (foreground + background * (1.0 - alpha)).clamp(0.0, 1.0)


def save_tensor_rgb(path: Path, tensor) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = tensor.detach().permute(1, 2, 0).cpu().numpy()
    bgr = np.clip(np.round(rgb[..., ::-1] * 255.0), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Could not save {path}")


def render_rotating_light(args: argparse.Namespace, input_ply: Path) -> None:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as functional

    sys.path.insert(0, str(args.repo))
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
    test_cameras = scene.getTestCameras()
    if max(args.view_indices) >= len(test_cameras):
        raise IndexError(
            f"--view-indices {args.view_indices} outside available range "
            f"[0, {len(test_cameras) - 1}]"
        )
    cameras = {index: test_cameras[index] for index in args.view_indices}
    pipe = pipeline_args()
    bg_color = [0, 0, 0]
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    object_gaussians = {
        "xyz": model._xyz,
        "features_dc": model._features_dc,
        "features_rest": model._features_rest,
        "opacity": model._opacity,
        "scaling": model._scaling,
        "rotation": model._rotation,
        "normal": None,
        "max_sh_degree": model.max_sh_degree,
    }
    base_dc = model._features_dc.detach().clone()
    base_rest = model._features_rest.detach().clone()
    normals = transfer.quaternion2rotmat(
        functional.normalize(model._rotation, dim=1)
    )[..., 2]
    normals = functional.normalize(normals, dim=1)
    device = "cuda"
    l_max = model.max_sh_degree

    print("Precomputing SH basis and baking visibility once...", flush=True)
    (
        directions,
        sqrt_weights,
        sh_basis,
        basis_weighted,
        ata_weighted,
        ata,
        gaunt_tensor,
    ) = transfer.precompute_sh_sampling(args.num_samples, l_max, device)
    ata_inverse = torch.linalg.inv(ata)
    with torch.no_grad():
        visibility_lm = transfer.phase_1_bake_visibility_sh(
            object_gaussians,
            directions,
            basis_weighted,
            ata_weighted,
            sqrt_weights,
            device,
            res=args.shadow_res,
        )

    source_bgr = cv2.imread(
        str(args.source_env), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
    )
    target_bgr = cv2.imread(
        str(args.target_env), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
    )
    if source_bgr is None or target_bgr is None:
        raise RuntimeError("Could not read source or target HDR")
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    source_gpu = torch.from_numpy(source_rgb).to(device)
    offset_gpu = torch.full_like(source_gpu, 0.5)
    with torch.no_grad():
        source_lm = transfer.compute_global_sh_coeffs(
            source_gpu, directions, sqrt_weights, basis_weighted, ata_weighted
        )
        offset_lm = transfer.compute_global_sh_coeffs(
            offset_gpu, directions, sqrt_weights, basis_weighted, ata_weighted
        )

    from tqdm import tqdm

    for index in tqdm(
        range(args.frames),
        desc=f"Rotating {args.target_name} relight + render",
    ):
        yaw = yaw_degrees(index, args.frames)
        rotated_bgr = rotate_envmap(target_bgr, yaw)
        save_hdr_and_preview(args, index, rotated_bgr)
        missing_views = [
            view
            for view in args.view_indices
            if args.force or not render_path(args, view, index).is_file()
        ]
        if not missing_views:
            continue

        target_rgb = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        target_for_sh = cv2.resize(
            target_rgb,
            (source_rgb.shape[1], source_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        target_gpu = torch.from_numpy(target_for_sh).to(device)
        with torch.no_grad():
            target_lm = transfer.compute_global_sh_coeffs(
                target_gpu,
                directions,
                sqrt_weights,
                basis_weighted,
                ata_weighted,
            )
            object_gaussians["features_dc"] = base_dc
            object_gaussians["features_rest"] = base_rest
            transfer.phase_2_decoupled_relight(
                object_gaussians,
                normals,
                visibility_lm,
                source_lm,
                target_lm,
                offset_lm,
                sh_basis,
                ata_inverse,
                gaunt_tensor,
                directions,
                l_max,
                device,
                floor_alpha=args.floor_alpha,
                tau_max=args.tau_max,
                specular_boost=args.specular_boost,
            )
            model._features_dc = object_gaussians["features_dc"]
            model._features_rest = object_gaussians["features_rest"]
            env_rgb = cv2.cvtColor(rotated_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            env_tensor = torch.from_numpy(env_rgb).to(device)

            for view_index in missing_views:
                camera = cameras[view_index]
                package = render(camera, model, pipe, background)
                composited = composite_envmap(
                    package["render"],
                    package["rend_alpha"],
                    camera,
                    env_tensor,
                    bg_color,
                )
                save_tensor_rgb(render_path(args, view_index, index), composited)


def fit_text(image, text, origin, max_width, scale, color, thickness=2) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_DUPLEX
    while scale > 0.35:
        (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if text_width <= max_width:
            break
        scale -= 0.05
    cv2.putText(
        image, text, origin, font, scale, color, thickness, cv2.LINE_AA
    )


def compose_frame(args: argparse.Namespace, view_index: int, index: int):
    import cv2
    import numpy as np

    rendered = cv2.imread(str(render_path(args, view_index, index)), cv2.IMREAD_COLOR)
    if rendered is None:
        raise RuntimeError(f"Missing source frame for view {view_index}, frame {index}")
    if args.video_layout == "render-only":
        return cv2.resize(
            rendered, (args.width, args.height), interpolation=cv2.INTER_LANCZOS4
        )

    panorama = cv2.imread(str(preview_path(args, index)), cv2.IMREAD_COLOR)
    if panorama is None:
        raise RuntimeError(f"Missing panorama for frame {index}")
    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    canvas[:] = (17, 14, 11)
    margin = max(32, args.width // 20)
    panel_width = args.width - 2 * margin
    pano_height = panel_width // 2
    panorama = cv2.resize(
        panorama, (panel_width, pano_height), interpolation=cv2.INTER_AREA
    )
    pano_y = 40
    canvas[pano_y : pano_y + pano_height, margin : margin + panel_width] = panorama

    render_y = pano_y + pano_height + 40
    available_height = args.height - render_y - 40
    render_side = min(panel_width, available_height)
    render_x = (args.width - render_side) // 2
    rendered = cv2.resize(
        rendered, (render_side, render_side), interpolation=cv2.INTER_LANCZOS4
    )
    canvas[
        render_y : render_y + render_side, render_x : render_x + render_side
    ] = rendered

    cv2.rectangle(
        canvas,
        (margin, pano_y),
        (margin + panel_width - 1, pano_y + pano_height - 1),
        (105, 100, 92),
        2,
    )
    cv2.rectangle(
        canvas,
        (render_x, render_y),
        (render_x + render_side - 1, render_y + render_side - 1),
        (105, 100, 92),
        2,
    )
    overlay = canvas.copy()
    cv2.rectangle(
        overlay, (margin, pano_y), (margin + panel_width, pano_y + 82), (10, 10, 10), -1
    )
    cv2.rectangle(
        overlay,
        (render_x, render_y),
        (render_x + render_side, render_y + 78),
        (10, 10, 10),
        -1,
    )
    cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0, canvas)

    yaw = yaw_degrees(index, args.frames)
    fit_text(
        canvas,
        f"ROTATING TARGET LIGHT  /  {args.target_name.upper()}  /  YAW {yaw:05.1f}",
        (margin + 28, pano_y + 53),
        panel_width - 56,
        0.92,
        (245, 245, 245),
    )
    fit_text(
        canvas,
        f"TRANSPLAT RELIT {args.object_name.upper()}  /  FIXED TEST VIEW {view_index:03d}",
        (render_x + 24, render_y + 51),
        render_side - 48,
        0.76,
        (245, 245, 245),
    )
    progress = int(round((index + 1) / args.frames * panel_width))
    cv2.rectangle(
        canvas,
        (margin, args.height - 18),
        (margin + panel_width, args.height - 13),
        (63, 59, 54),
        -1,
    )
    cv2.rectangle(
        canvas,
        (margin, args.height - 18),
        (margin + progress, args.height - 13),
        (235, 235, 235),
        -1,
    )
    return canvas


def encode_view_video(args: argparse.Namespace, view_index: int) -> tuple[Path, Path]:
    import cv2
    from tqdm import tqdm

    video_dir = args.output_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.object_name}_{args.target_name}_rotating_light_test_{view_index:03d}"
    video_path = video_dir / f"{stem}.mp4"
    poster_path = video_dir / f"{stem}_poster.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for index in tqdm(range(args.frames), desc=f"Encode test view {view_index:03d}"):
            frame = compose_frame(args, view_index, index)
            if index == args.frames // 8:
                cv2.imwrite(
                    str(poster_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94]
                )
            process.stdin.write(frame.tobytes())
    except BrokenPipeError as error:
        raise RuntimeError("ffmpeg stopped while receiving frames") from error
    finally:
        process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed for test view {view_index}")
    return video_path, poster_path


def write_manifest(
    args: argparse.Namespace, videos: dict[int, tuple[Path, Path]]
) -> Path:
    manifest = {
        "object": args.object_name,
        "source_environment": str(args.source_env),
        "rotating_target_environment": str(args.target_env),
        "video_layout": args.video_layout,
        "rotation_convention": "positive horizontal np.roll",
        "frames": args.frames,
        "yaw_step_degrees": 360.0 / args.frames,
        "fps": args.fps,
        "test_view_indices": args.view_indices,
        "rotated_hdr_directory": str(args.output_dir / "rotated_envmaps" / "hdr"),
        "render_directories": {
            f"{view:03d}": str(args.output_dir / "renders" / f"test_{view:03d}")
            for view in args.view_indices
        },
        "videos": {
            f"{view:03d}": {"video": str(paths[0]), "poster": str(paths[1])}
            for view, paths in videos.items()
        },
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main() -> int:
    args = parse_args()
    reexec_in_conda(args)
    input_ply = validate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', '(unknown)')}")
    print(f"Test views: {', '.join(f'{view:03d}' for view in args.view_indices)}")
    print(f"Yaw frames: {args.frames} ({360.0 / args.frames:.4f} degrees/frame)")

    complete = outputs_complete(args)
    if args.skip_relight_render:
        if not complete:
            raise RuntimeError("Existing HDR/render frame set is incomplete")
    elif complete and not args.force:
        print("[reuse] All rotated HDR and fixed-view render frames are complete")
    else:
        render_rotating_light(args, input_ply)

    if not outputs_complete(args):
        raise RuntimeError("Rotating-light output frame set is incomplete")

    videos: dict[int, tuple[Path, Path]] = {}
    if not args.skip_video:
        for view_index in args.view_indices:
            videos[view_index] = encode_view_video(args, view_index)
    manifest_path = write_manifest(args, videos)

    print("\nComplete")
    for view_index, (video, _) in videos.items():
        print(f"View {view_index:03d}: {video}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
