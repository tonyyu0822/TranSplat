"""Physical DC lighting utilities: SH evaluation, prior loading, source-SH solving."""

import math
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# ── SH constants ─────────────────────────────────────────────────────────────
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
       0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]

# Lambertian convolution kernel A_l for l=0..3 (16 coefficients)
LAMBERT_A_L_16 = torch.tensor([
    math.pi,
    2*math.pi/3, 2*math.pi/3, 2*math.pi/3,
    math.pi/4, math.pi/4, math.pi/4, math.pi/4, math.pi/4,
    0., 0., 0., 0., 0., 0., 0.
], dtype=torch.float32)


# ── SH evaluation ─────────────────────────────────────────────────────────────
def evaluate_sh_bases(directions: torch.Tensor, l_max: int) -> torch.Tensor:
    """directions: [N,3] unit vectors  →  [N, (l_max+1)^2] SH basis values."""
    device = directions.device
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    bases = [torch.full((len(x),), C0, device=device, dtype=directions.dtype)]
    if l_max >= 1:
        bases += [-C1*y, C1*z, -C1*x]
        if l_max >= 2:
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z
            bases += [C2[0]*xy, C2[1]*yz, C2[2]*(2*zz-xx-yy), C2[3]*xz, C2[4]*(xx-yy)]
            if l_max >= 3:
                bases += [
                    C3[0]*y*(3*xx-yy), C3[1]*xy*z, C3[2]*y*(4*zz-xx-yy),
                    C3[3]*z*(2*zz-3*xx-3*yy), C3[4]*x*(4*zz-xx-yy),
                    C3[5]*z*(xx-yy), C3[6]*x*(xx-3*yy),
                ]
    return torch.stack(bases, dim=1)


# ── Colour-space helpers ───────────────────────────────────────────────────────
def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92,
                       torch.pow((c + 0.055) / 1.055, 2.4))


def linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.0031308, c * 12.92,
                       1.055 * torch.pow(c.clamp(min=1e-10), 1.0 / 2.4) - 0.055)


# ── Sampling matrix for non-negativity check ──────────────────────────────────
def make_sh_sample_matrix(num_samples: int = 256, l_max: int = 3,
                           device: str = 'cuda') -> torch.Tensor:
    """Fibonacci-lattice SH basis [num_samples, (l_max+1)^2] for non-neg check."""
    idx = torch.arange(num_samples, device=device, dtype=torch.float64)
    theta = torch.acos((1 - 2*(idx+0.5)/num_samples).clamp(-1, 1))
    phi   = math.pi * (1 + 5**0.5) * idx
    dirs  = torch.stack([
        torch.sin(theta)*torch.cos(phi),
        torch.sin(theta)*torch.sin(phi),
        torch.cos(theta),
    ], dim=1).float()
    return evaluate_sh_bases(dirs, l_max)


# ── Per-camera prior loading ───────────────────────────────────────────────────
_PRIOR_CACHE: dict = {}


def load_prior(cam, device: str = 'cuda'):
    """Return (albedo [3,H,W] sRGB∈[0,1], normal [3,H,W] unit, mask [1,H,W])
    or None if this camera has no prior data. Results are cached."""
    uid = cam.uid
    if uid in _PRIOR_CACHE:
        return _PRIOR_CACHE[uid]

    # Derive frame directory from image_path stored on Camera
    image_path = getattr(cam, 'image_path', None)
    if image_path is None:
        _PRIOR_CACHE[uid] = None
        return None

    frame_dir   = os.path.dirname(image_path)
    albedo_path = os.path.join(frame_dir, 'albedo.png')
    normal_path = os.path.join(frame_dir, 'normal.exr')

    if not (os.path.exists(albedo_path) and os.path.exists(normal_path)):
        _PRIOR_CACHE[uid] = None
        return None

    # Albedo: sRGB uint8 → float [0,1]
    alb_bgr = cv2.imread(albedo_path, cv2.IMREAD_COLOR)
    alb = torch.from_numpy(
        cv2.cvtColor(alb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ).permute(2, 0, 1).to(device)          # [3, H, W]

    # Normal: world-space float, convert BGR→RGB
    n_bgr = cv2.imread(normal_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if n_bgr is None:
        _PRIOR_CACHE[uid] = None
        return None
    n_rgb = cv2.cvtColor(n_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    norm  = torch.from_numpy(n_rgb).permute(2, 0, 1).to(device)  # [3, H, W]
    norm  = F.normalize(norm, dim=0)

    # Mask: non-background (sum of albedo channels > threshold)
    mask = (alb.sum(0, keepdim=True) > 0.02).float()              # [1, H, W]

    result = (alb, norm, mask)
    _PRIOR_CACHE[uid] = result
    return result


# ── Source-SH initialisation ──────────────────────────────────────────────────
@torch.no_grad()
def solve_source_sh(train_cameras, gaussians, pipe, bg_color,
                    l_max: int = 3, max_cams: int = 20,
                    device: str = 'cuda') -> torch.Tensor:
    """Solve for source_sh [K,3] using GT normals, GT albedo and current DC renders.
    Pixel-wise LS: Y_hat @ L ≈ dc_linear / albedo_linear."""
    from gaussian_renderer import render

    A_l   = LAMBERT_A_L_16[:(l_max+1)**2].to(device)
    K     = (l_max + 1) ** 2
    Y_all, E_all = [], []

    import random
    cams = random.sample(train_cameras, min(max_cams, len(train_cameras)))

    for cam in cams:
        prior = load_prior(cam, device)
        if prior is None:
            continue
        alb_gt, norm_gt, mask_gt = prior          # [3,H,W], [3,H,W], [1,H,W]

        pkg = render(cam, gaussians, pipe, bg_color)
        rendered = pkg['render'].clamp(0, 1)       # [3,H,W] sRGB

        H, W = alb_gt.shape[1], alb_gt.shape[2]
        rend_r = F.interpolate(rendered.unsqueeze(0), (H, W), mode='bilinear',
                               align_corners=False)[0]

        rend_lin = srgb_to_linear(rend_r)
        alb_lin  = srgb_to_linear(alb_gt.clamp(0, 1))
        E_est    = rend_lin / (alb_lin + 1e-3)    # [3,H,W]

        m = (mask_gt[0] > 0.5)                    # [H,W] bool
        n_flat = norm_gt.permute(1, 2, 0)[m]      # [K, 3]
        E_flat = E_est.permute(1, 2, 0)[m]        # [K, 3]

        if n_flat.shape[0] > 8000:
            idx = torch.randperm(n_flat.shape[0], device=device)[:8000]
            n_flat, E_flat = n_flat[idx], E_flat[idx]

        n_sh = n_flat.clone(); n_sh[:, 0] *= -1   # x-flip: world→SH convention
        n_sh = F.normalize(n_sh, dim=1)
        Y     = evaluate_sh_bases(n_sh, l_max)    # [K, 16]
        Y_hat = Y * A_l.view(1, -1)               # Lambertian-weighted

        Y_all.append(Y_hat); E_all.append(E_flat)

    if not Y_all:
        # Fallback: neutral ambient
        L = torch.zeros(K, 3, device=device)
        L[0] = 0.5 / math.pi          # E ≈ π * L0 * C0 ≈ 0.5
        return L

    Y_mat = torch.cat(Y_all, dim=0)               # [total, K]
    E_mat = torch.cat(E_all, dim=0)               # [total, 3]

    lhs = Y_mat.T @ Y_mat + 1e-3 * torch.eye(K, device=device)
    rhs = Y_mat.T @ E_mat
    L   = torch.linalg.solve(lhs, rhs)            # [K, 3]
    print(f"[source_sh init] L00={L[0].tolist()}  "
          f"relerr={((Y_mat@L - E_mat).norm() / E_mat.norm()).item():.3f}")
    return L.detach()


# ── Debug visualisation ────────────────────────────────────────────────────────
def save_debug_vis(output_dir: str, iteration: int,
                   render_pkg: dict, gaussians,
                   gt_image: torch.Tensor,
                   gt_albedo=None, gt_normal=None,
                   albedo_render=None):
    vis_dir = os.path.join(output_dir, 'debug_vis')
    os.makedirs(vis_dir, exist_ok=True)

    def _save(name, t):
        """t: [3,H,W] or [1,H,W] float tensor in [0,1]."""
        if t is None:
            return
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        arr = (t.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255
               ).astype(np.uint8)
        path = os.path.join(vis_dir, f'iter{iteration:05d}_{name}.png')
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    _save('render',      render_pkg['render'])
    _save('gt',          gt_image)
    _save('rend_normal', (render_pkg['rend_normal'] + 1) * 0.5)
    _save('surf_normal', (render_pkg['surf_normal'] + 1) * 0.5)
    _save('rend_alpha',  render_pkg.get('rend_alpha'))

    if gt_albedo  is not None: _save('gt_albedo',  gt_albedo)
    if gt_normal  is not None: _save('gt_normal',  (gt_normal + 1) * 0.5)
    if albedo_render is not None: _save('pred_albedo', albedo_render)

    # Source SH preview for older checkpoints that carried physical-DC state.
    source_sh = getattr(gaussians, "_source_sh", None)
    if getattr(gaussians, "_use_physical_dc", False) and source_sh is not None and source_sh.numel() > 0:
        _write_sh_preview(source_sh.detach(),
                          os.path.join(vis_dir, f'iter{iteration:05d}_source_sh_env.png'))

    print(f'[vis] saved iter {iteration} → {vis_dir}')


def _write_sh_preview(L: torch.Tensor, path: str, h: int = 64, w: int = 128):
    """Render a tiny equirectangular env-map preview from SH coefficients L [K,3]."""
    device = L.device
    l_max  = int(math.sqrt(L.shape[0]) - 1)
    theta  = torch.linspace(0, math.pi,   h, device=device)
    phi    = torch.linspace(0, 2*math.pi, w, device=device)
    T, P   = torch.meshgrid(theta, phi, indexing='ij')
    dirs   = torch.stack([
        torch.sin(T)*torch.cos(P),
        torch.sin(T)*torch.sin(P),
        torch.cos(T),
    ], dim=-1).reshape(-1, 3)
    img = (evaluate_sh_bases(dirs, l_max) @ L).reshape(h, w, 3).clamp(0)
    img_u8 = (img / img.max().clamp(min=1e-6) * 255).byte().cpu().numpy()
    cv2.imwrite(path, cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
