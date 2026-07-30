#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import uuid
import torch
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr, render_net_image
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer  = prepare_output_and_logger(dataset)
    gaussians  = GaussianModel(dataset.sh_degree)
    scene      = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    train_cameras = scene.getTrainCameras()

    viewpoint_stack = None
    ema_loss = ema_dist = ema_normal = ema_alpha = 0.0
    psnr_history = []           # (iter, psnr) for monitoring

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if opt.geometry_freeze_iter > 0 and iteration >= opt.geometry_freeze_iter:
            for param_group in gaussians.optimizer.param_groups:
                if param_group["name"] in ("xyz", "rotation", "scaling"):
                    param_group["lr"] = 0.0

        if iteration % opt.sh_degree_up_interval == 0:
            gaussians.oneupSHdegree()

        # ── Pick a random camera ──────────────────────────────────────────────
        if not viewpoint_stack:
            viewpoint_stack = list(train_cameras)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # ── Forward render ────────────────────────────────────────────────────
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter      = render_pkg["visibility_filter"]
        radii                  = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        # ── Photometric loss ──────────────────────────────────────────────────
        Ll1  = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1 - ssim(image, gt_image))

        # ── Vanilla 2DGS regularisers ─────────────────────────────────────────
        if iteration < opt.normal_consistency_ramp_start:
            lambda_normal = 0.0
        elif iteration < opt.normal_consistency_ramp_end:
            progress = (iteration - opt.normal_consistency_ramp_start) / (
                opt.normal_consistency_ramp_end - opt.normal_consistency_ramp_start)
            lambda_normal = opt.lambda_normal * progress
        else:
            lambda_normal = opt.lambda_normal
        lambda_dist   = opt.lambda_dist   if iteration > 3000 else 0.0

        rend_dist   = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * normal_error.mean()
        loss += normal_loss
        loss += lambda_dist   * rend_dist.mean()

        alpha_loss = torch.tensor(0.0, device="cuda")
        if viewpoint_cam.gt_alpha_mask is not None and opt.lambda_alpha > 0:
            gt_alpha = viewpoint_cam.gt_alpha_mask.cuda()
            rend_alpha = render_pkg["rend_alpha"]
            alpha_loss = opt.lambda_alpha * l1_loss(rend_alpha, gt_alpha)
            loss += alpha_loss

        # ── Backward ─────────────────────────────────────────────────────────
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss   = 0.4 * loss.item() + 0.6 * ema_loss
            ema_dist   = 0.4 * rend_dist.mean().item() + 0.6 * ema_dist
            ema_normal = 0.4 * normal_loss.item() + 0.6 * ema_normal
            ema_alpha  = 0.4 * alpha_loss.item() + 0.6 * ema_alpha

            # ── Progress bar ──────────────────────────────────────────────────
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss:.4f}",
                    "dist": f"{ema_dist:.5f}",
                    "norm": f"{ema_normal:.5f}",
                    "alpha": f"{ema_alpha:.5f}",
                    "N":    len(gaussians.get_xyz),
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # ── TensorBoard ───────────────────────────────────────────────────
            if tb_writer:
                tb_writer.add_scalar('loss/total',  loss.item(), iteration)
                tb_writer.add_scalar('loss/l1',     Ll1.item(),  iteration)
                tb_writer.add_scalar('loss/dist',   ema_dist,    iteration)
                tb_writer.add_scalar('loss/normal', ema_normal,  iteration)
                tb_writer.add_scalar('loss/alpha',  ema_alpha,   iteration)

            # ── Save & eval ───────────────────────────────────────────────────
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background),
                            psnr_history)
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # ── Densification ─────────────────────────────────────────────────
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and \
                        iteration % opt.densification_interval == 0:
                    size_thresh = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opt.opacity_cull,
                        scene.cameras_extent, size_thresh)

                if iteration % opt.opacity_reset_interval == 0 or \
                        (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # ── Optimiser step ────────────────────────────────────────────────
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + f"/chkpnt{iteration}.pth")

# ── Logger & report (unchanged from vanilla) ─────────────────────────────────
def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID', str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str[:10])
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene, renderFunc, renderArgs, psnr_history):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss',   Ll1.item(),   iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(),  iteration)
        tb_writer.add_scalar('iter_time',    elapsed,    iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        configs = (
            {'name': 'test',  'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[i % len(scene.getTrainCameras())]
                                          for i in range(5, 30, 5)]},
        )
        for config in configs:
            if not config['cameras']:
                continue
            l1_t = psnr_t = 0.0
            for idx, vp in enumerate(config['cameras']):
                pkg   = renderFunc(vp, scene.gaussians, *renderArgs)
                img   = torch.clamp(pkg["render"], 0.0, 1.0).cuda()
                gt    = torch.clamp(vp.original_image.cuda(), 0.0, 1.0)
                if tb_writer and idx < 5:
                    from utils.general_utils import colormap
                    depth = pkg["surf_depth"]
                    depth = colormap(( depth / depth.max()).cpu().numpy()[0], cmap='turbo')
                    tb_writer.add_images(f'{config["name"]}_view_{vp.image_name}/depth',
                                         depth[None], global_step=iteration)
                    tb_writer.add_images(f'{config["name"]}_view_{vp.image_name}/render',
                                         img[None], global_step=iteration)
                    try:
                        rn = pkg["rend_normal"] * 0.5 + 0.5
                        sn = pkg["surf_normal"] * 0.5 + 0.5
                        tb_writer.add_images(f'{config["name"]}_view_{vp.image_name}/rend_normal',
                                             rn[None], global_step=iteration)
                        tb_writer.add_images(f'{config["name"]}_view_{vp.image_name}/surf_normal',
                                             sn[None], global_step=iteration)
                    except Exception:
                        pass
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(f'{config["name"]}_view_{vp.image_name}/gt',
                                             gt[None], global_step=iteration)
                l1_t   += l1_loss(img, gt).mean().double()
                psnr_t += psnr(img, gt).mean().double()

            psnr_t /= len(config['cameras'])
            l1_t   /= len(config['cameras'])
            print(f"\n[ITER {iteration}] Eval {config['name']}: L1={l1_t:.4f}  PSNR={psnr_t:.2f}")
            if tb_writer:
                tb_writer.add_scalar(f"{config['name']}/l1",   l1_t,   iteration)
                tb_writer.add_scalar(f"{config['name']}/psnr", psnr_t, iteration)
            if config['name'] == 'test':
                psnr_history.append((iteration, float(psnr_t)))

        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip',   type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations",       nargs="+", type=int,
                        default=[7000, 30000])
    parser.add_argument("--save_iterations",       nargs="+", type=int,
                        default=[7000, 30000])
    parser.add_argument("--quiet",                 action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint",      type=str,  default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint)
    print("\nTraining complete.")
