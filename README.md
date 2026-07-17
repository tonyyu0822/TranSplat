<p align="center">
  <img src="assets/logo.png" alt="TranSplat" width="360">
</p>

# TranSplat: Instant Cross-Scene Object Relighting in Gaussian Splatting via Spherical Harmonic Transfer

<p align="center"><b>ICCP 2026</b></p>

[Project Page](https://tonyyu0822.github.io/transplat/) | [Paper](https://arxiv.org/abs/2503.22676) | [arXiv](https://arxiv.org/abs/2503.22676) | [2D Gaussian Splatting (base)](https://github.com/hbb1/2d-gaussian-splatting) <br>

![Teaser: before/after TranSplat relighting on the bonsai and vase scenes](assets/teaser.gif)

This repo contains the official implementation for **TranSplat**, a method for instant, accurate object relighting within the Gaussian Splatting (GS) framework. Rather than relying on costly inverse-rendering routines, TranSplat is a BRDF-free radiance transfer strategy that analytically modulates the spherical harmonic (SH) appearance coefficients of an object's 2D Gaussian surfels using per-normal irradiance ratios derived from source and target environment maps. A specularity-aware dual-path SH transfer strategy adapts higher-order SH bands in the reflection domain for view-dependent, glossy appearance, and a lightweight SH-domain self-shadowing module produces physically realistic occlusion without explicit mesh raycasting. TranSplat is a post-processing step — it requires no additional GS retraining per source/target scene pair, and it relights in well under one second.

Built on top of [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting), which represents a scene as a set of 2D oriented disks (surfels) rasterized with perspective-correct differentiable rasterization. **The core of TranSplat is a single, self-contained file — [`radiance_transfer_TranSplat.py`](radiance_transfer_TranSplat.py)** — everything else in this repo is the (largely unmodified) 2DGS training/rendering pipeline plus a few diagnostic/demo scripts around it.

## ⭐ News
- 2026/07/17: Initial public code release, alongside the [project page](https://tonyyu0822.github.io/transplat/).

## 🚧 GUI (Coming Soon)

We are packaging the interactive GUI shown below — real-time source/target environment-map sampling with live, spatially-varying relighting — for public release. It is **not** part of this repository yet; the preview below is a capture of the internal tool.

![Preview of the upcoming interactive relighting GUI — not yet part of this repository](assets/gui_preview.gif)

## Method

![TranSplat method overview](assets/method.png)

## Installation

```bash
# download
git clone --recursive https://github.com/tonyyu0822/TranSplat.git
cd TranSplat

# if you already have a working 2D Gaussian Splatting conda environment,
# just activate that one and skip straight to `pip install` below —
# TranSplat adds no new dependencies on top of 2DGS.
conda env create --file environment.yml
conda activate surfel_splatting
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
```

**Custom data**: TranSplat uses the same COLMAP / NeRF-Synthetic loaders as 3DGS/2DGS. To prepare your own COLMAP scene, see `convert.py` and the [3DGS data prep instructions](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes).

## Training

To train a scene, simply use
```bash
python train.py -s <path to COLMAP or NeRF Synthetic dataset> -m <output dir>
```
or, via the driver script (see [Running the pipeline](#running-the-pipeline) below):
```bash
scripts/run_transplat.sh train -s <dataset> -m <output dir>
```

Commandline arguments for regularization:
```bash
--lambda_normal              # hyperparameter for normal consistency
--lambda_dist                # hyperparameter for depth distortion
--depth_ratio                # 0 for mean depth, 1 for median depth (0 works for most cases)
--sh_degree                  # max SH degree (default: 3)
--sh_degree_up_interval      # increase active SH degree by 1 every N iterations (default: 1000)
```
Higher final SH quality on glossy/specular objects (important for relighting) generally benefits from a higher `--sh_degree` paired with a longer `--sh_degree_up_interval`, e.g.:
```bash
python train.py -s <dataset> -m <output dir> \
    --sh_degree 5 --sh_degree_up_interval 3000 \
    --lambda_normal 0.05 --lambda_dist 0.0
```
**Trade-off:** increasing `--sh_degree` improves relight quality — especially specular highlights — since TranSplat's specular transfer operates directly on the higher-order SH bands, but it also increases per-Gaussian storage and rendering cost, lowering FPS. See the paper for a detailed quality/speed analysis.

The rest of the physically-based DC/albedo prior schedule (used to keep SH coefficients shadow-free and relight-ready) is controlled by constants near the top of `train.py` for advanced tuning.

## Relighting

Relighting is a post-training, post-processing step: it reads a trained point cloud and a pair of HDR environment maps, and writes out a new, relit point cloud — no retraining required.

```bash
python radiance_transfer_TranSplat.py \
    --source <source.hdr> --target <target.hdr> \
    --ply <path to trained point_cloud.ply> \
    --output <path to write the relit .ply>
```
or
```bash
scripts/run_transplat.sh relight --ply <point_cloud.ply> --source-env <source.hdr> --target-env <target.hdr> -o <relit.ply>
```

Useful flags: `--floor_alpha` (diffuse denominator floor), `--tau_max` (diffuse transfer clamp), `--specular_boost` (scale for L≥1 SH rest bands), `--num_samples`/`--shadow_res` (self-shadow bake quality/cost trade-off).

<table>
<tr>
<td width="50%"><img src="assets/relight_multienv_loop.gif" alt="TranSplat relighting the same object across several environment maps"></td>
<td width="50%"><img src="assets/rotating_light_comparison.gif" alt="Spatially-varying shadows and relighting as the target environment map rotates"></td>
</tr>
<tr>
<td align="center">Relighting across multiple target environment maps</td>
<td align="center">Spatially-varying shadows &amp; relighting under a rotating light</td>
</tr>
</table>

## Rendering / Mesh Extraction

Render a trained (or relit) point cloud and/or extract a mesh:
```bash
# Render the trained checkpoint and extract a bounded mesh
python render.py -m <path to trained model> -s <path to COLMAP dataset>

# Render a specific point cloud instead (e.g. the relit .ply from above)
python render.py -m <path to trained model> -s <path to dataset> --ply <relit.ply> --skip_mesh
```
or
```bash
scripts/run_transplat.sh render -s <dataset> -m <model dir> --ply <relit.ply> --skip-mesh
```

Meshing arguments (same as upstream 2DGS):
```bash
--depth_ratio     # 0 for mean depth, 1 for median depth
--voxel_size      # voxel size for TSDF fusion
--depth_trunc     # depth truncation
--unbounded       # unbounded mesh extraction (space contraction + adaptive TSDF truncation)
--mesh_res        # resolution for unbounded mesh extraction
```
If unspecified, meshing arguments are estimated automatically from the camera information.

## Running the pipeline

`scripts/run_transplat.sh` is a single entry point for training, relighting, and rendering. Each subcommand prints a stage banner and a clear `OK`/`FAILED` result; `pipeline` chains train → relight → render for one scene and stops immediately — naming the failed stage — if anything goes wrong.

```bash
# One-shot: train, relight against a target envmap, and render the result
scripts/run_transplat.sh pipeline \
    -s <dataset> -m <output dir> \
    --source-env <source.hdr> --target-env <target.hdr>

# Individual stages
scripts/run_transplat.sh train      -s <dataset> -m <output dir>
scripts/run_transplat.sh relight    --ply <point_cloud.ply> --source-env <src.hdr> --target-env <tgt.hdr> -o <relit.ply>
scripts/run_transplat.sh render     -s <dataset> -m <output dir> --ply <relit.ply> --skip-mesh
```
Every subcommand accepts extra flags forwarded verbatim to the underlying Python script after a literal `--`, e.g. `scripts/run_transplat.sh train -s <dataset> -m <out> -- --test_iterations 5000 10000`. Run any subcommand with `-h` for its full option list.

## Diagnostics & Demos

Two additional scripts, also reachable through `run_transplat.sh`:

- **Directional visibility diagnostic** — bakes the self-shadowing visibility function and renders it as an RGB image (R/G/B = visibility from world +X/+Y/+Z), useful for inspecting the shadow bake independently of any relighting pair.
  ```bash
  scripts/run_transplat.sh visibility --dataset <dataset> --model <model dir>
  ```
- **Rotating-envmap demo** — bakes visibility once, then relights and renders fixed camera views while the target environment map rotates around the object, encoding the result as an MP4 (this is what produced the "rotating light" comparison GIF in the [Relighting](#relighting) section above).
  ```bash
  scripts/run_transplat.sh demo --dataset <dataset> --model <model dir> --source-env <src.hdr> --target-env <tgt.hdr>
  ```

## Results

![Qualitative comparison against recent Gaussian relighting methods](assets/experiments_comparison.jpg)

TranSplat outperforms recent inverse-rendering and diffusion-based GS relighting methods across most conditions on synthetic and real-world objects, while completing relighting in under one second. See the [paper](https://arxiv.org/abs/2503.22676) for full quantitative results.

## FAQ
- **Training does not converge.** If your camera's principal point does not lie at the image center, you may experience convergence issues — this codebase only supports the ideal pinhole camera format. See [this note](https://github.com/graphdeco-inria/gaussian-splatting/issues/144#issuecomment-1938504456) from the 3DGS repo for the required fix.
- **No mesh / broken mesh.** In bounded mesh extraction, adjust `--depth_trunc` to perform TSDF fusion correctly. Unbounded mesh extraction (`--unbounded`) does not require tuning but is less efficient.
- **Relighting looks over/under-saturated.** Try adjusting `--tau_max` (clamps the diffuse transfer ratio) and `--specular_boost` (scales specular SH rest bands) in `radiance_transfer_TranSplat.py`.
- **Where's the GUI?** See [GUI (Coming Soon)](#-gui-coming-soon) above — it's being packaged for release separately from this repo.

## Acknowledgements
This project is built upon [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) (Huang et al., SIGGRAPH 2024), which itself builds on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). The TSDF fusion for mesh extraction is based on [Open3D](https://github.com/isl-org/Open3D). We thank the authors of both projects for their work.

## Citation
If you find TranSplat useful for your work, please consider citing:
```bibtex
@misc{yu2025transplatinstantcrosssceneobject,
      title={TranSplat: Instant Cross-Scene Object Relighting in Gaussian Splatting via Spherical Harmonic Transfer},
      author={Boyang Yu and Yanlin Jin and Yun He and Akshat Dave and Ravi Ramamoorthi and Guha Balakrishnan},
      year={2025},
      eprint={2503.22676},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.22676},
}
```
