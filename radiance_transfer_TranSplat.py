import torch
import numpy as np
from torch import nn
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import os
import time
import math
import argparse
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# ============================================================================
# SH Constants
# ============================================================================
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
C4 = [2.5033429417967046, -1.7701307697799304, 0.9461746957575601, -0.6690465435572892, 0.10578554691520431, -0.6690465435572892, 0.47308734787878004, -1.7701307697799304, 0.6258357354491761]

# ============================================================================
# Configuration: File Paths
# ============================================================================
ENV_MAP_SOURCE_PATH = '/scratch/shared/by12/Transplat/envmap/city.hdr'
ENV_MAP_TARGET_PATH = '/scratch/shared/by12/Transplat/envmap/px.hdr'
OBJECT_PATH = '/scratch/shared/by12/Transplat/code/2dgs_viewer/2d-gaussian-splatting/output/armadillo_vanilla/point_cloud/iteration_30000/point_cloud.ply'
RELIGHT_OUTPUT_PATH = "/scratch/shared/by12/Transplat/code/2dgs_viewer/2d-gaussian-splatting/output/armadillo_vanilla/point_cloud/iteration_30000/city2px_point_cloud.ply"

# ============================================================================
# 2DGS Mock Classes & Camera Helpers
# ============================================================================
class MockGaussianModel:
    def __init__(self, gaussians_dict):
        self.g = gaussians_dict
        self.max_sh_degree = gaussians_dict['max_sh_degree']
        self.active_sh_degree = gaussians_dict['max_sh_degree']
    @property
    def get_xyz(self): return self.g['xyz']
    @property
    def get_features(self): return torch.cat([self.g['features_dc'], self.g['features_rest']], dim=1)
    @property
    def get_opacity(self): return torch.sigmoid(self.g['opacity'])
    @property
    def get_scaling(self): return torch.exp(self.g['scaling'])
    @property
    def get_rotation(self): return F.normalize(self.g['rotation'], dim=1)
    
class MockPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    depth_ratio = 0.0

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY, tanHalfFovX = math.tan((fovY / 2)), math.tan((fovX / 2))
    top, right = tanHalfFovY * znear, tanHalfFovX * znear
    bottom, left = -top, -right
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P

class MiniCam:
    def __init__(self, W, H, fovX, fovY, znear, zfar, W2C, full_proj, origin):
        self.image_width, self.image_height = W, H
        self.FoVx, self.FoVy = fovX, fovY
        self.znear, self.zfar = znear, zfar
        self.world_view_transform, self.full_proj_transform = W2C, full_proj
        self.camera_center = origin

def create_telephoto_camera(scene_center, direction, radius, W, H, device):
    D = radius * 5.0
    origin = scene_center + direction * D
    forward = F.normalize(scene_center - origin, dim=0)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    if abs(torch.dot(forward, up)) > 0.99: up = torch.tensor([1.0, 0.0, 0.0], device=device)
    right = F.normalize(torch.cross(up, forward), dim=0)
    true_up = F.normalize(torch.cross(forward, right), dim=0)
    R = torch.stack([right, -true_up, forward], dim=1)
    t = -origin @ R
    W2C = torch.eye(4, device=device)
    W2C[:3, :3], W2C[3, :3] = R, t
    fovY = 2.0 * math.asin(radius / D) * 1.2
    fovX = fovY
    znear, zfar = D - radius * 1.5, D + radius * 1.5
    P = getProjectionMatrix(znear, zfar, fovX, fovY).transpose(0, 1).to(device)
    return MiniCam(W, H, fovX, fovY, znear, zfar, W2C, W2C @ P, origin), W2C, W2C @ P

# ============================================================================
# PLY I/O Functions
# ============================================================================
def load_ply(path, normal_b=True):
    def sanitize(name, arr):
        bad = ~np.isfinite(arr)
        if bad.any():
            print(f"[WARN] {os.path.basename(path)}: replacing {int(bad.sum())} non-finite values in {name}")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])),  axis=1)
    normal = np.stack((np.asarray(plydata.elements[0]["nx"]), np.asarray(plydata.elements[0]["ny"]), np.asarray(plydata.elements[0]["nz"])), axis=1) if normal_b else None
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    for i in range(3): features_dc[:, i, 0] = np.asarray(plydata.elements[0][f"f_dc_{i}"])

    extra_f_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")], key = lambda x: int(x.split('_')[-1]))
    max_sh_degree = int(math.sqrt(len(extra_f_names) // 3 + 1)) - 1
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names): features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")], key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names): scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("rot")], key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names): rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = sanitize("xyz", xyz)
    if normal is not None:
        normal = sanitize("normal", normal)
    opacities = sanitize("opacity", opacities)
    features_dc = sanitize("features_dc", features_dc)
    features_extra = sanitize("features_rest", features_extra)
    scales = sanitize("scaling", scales)
    rots = sanitize("rotation", rots)

    return {
        "xyz": nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda")),
        "features_dc": nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()),
        "features_rest": nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()),
        "opacity": nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda")),
        "scaling": nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda")),
        "rotation": nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda")),
        "normal": nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda")) if normal_b else None,
        "max_sh_degree": max_sh_degree
    }

def save_ply_from_dict(dict, save_path):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(dict['features_dc'].shape[1]*dict['features_dc'].shape[2]): l.append(f'f_dc_{i}')
    for i in range(dict['features_rest'].shape[1]*dict['features_rest'].shape[2]): l.append(f'f_rest_{i}')
    l.append('opacity')
    for i in range(dict['scaling'].shape[1]): l.append(f'scale_{i}')
    for i in range(dict['rotation'].shape[1]): l.append(f'rot_{i}')
    
    xyz = dict['xyz'].detach().cpu().numpy()
    attributes = np.concatenate((
        xyz, np.zeros_like(xyz),
        dict['features_dc'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
        dict['features_rest'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
        dict['opacity'].detach().cpu().numpy(), dict['scaling'].detach().cpu().numpy(), dict['rotation'].detach().cpu().numpy()
    ), axis=1)
    
    elements = np.empty(xyz.shape[0], dtype=[(attr, 'f4') for attr in l])
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(save_path)

def quaternion2rotmat(q):
    r, x, y, z = q.split(1, -1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)
    ], -1).reshape([len(q), 3, 3])

# ============================================================================
# GPU-Optimized SH Functions
# ============================================================================
def evaluate_sh_bases_torch(directions, l_max, device):
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    bases = [torch.full((directions.shape[0],), C0, device=device, dtype=directions.dtype)]
    if l_max > 0:
        bases.extend([-C1 * y, C1 * z, -C1 * x])
        if l_max > 1:
            xx, yy, zz, xy, yz, xz = x*x, y*y, z*z, x*y, y*z, x*z
            bases.extend([C2[0]*xy, C2[1]*yz, C2[2]*(2.0*zz-xx-yy), C2[3]*xz, C2[4]*(xx-yy)])
            if l_max > 2:
                bases.extend([C3[0]*y*(3*xx-yy), C3[1]*xy*z, C3[2]*y*(4*zz-xx-yy), C3[3]*z*(2*zz-3*xx-3*yy), C3[4]*x*(4*zz-xx-yy), C3[5]*z*(xx-yy), C3[6]*x*(xx-3*yy)])
    return torch.stack(bases, dim=1) 

def precompute_sh_sampling(num_samples, l_max, device):
    indices = torch.arange(num_samples, device=device, dtype=torch.float64)
    theta = torch.acos(1 - 2 * (indices + 0.5) / num_samples)
    phi = torch.pi * (1 + 5 ** 0.5) * indices
    
    directions = torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)], dim=1).float()
    weights = torch.sin(theta).float()
    sqrt_weights = torch.sqrt(weights)
    
    A = evaluate_sh_bases_torch(directions, l_max, device)
    A_weighted = A * sqrt_weights.unsqueeze(1)
    AT_A_weighted = A_weighted.T @ A_weighted
    AT_A = A.T @ A
    
    dw_factor = (4.0 * torch.pi) / weights.sum()
    solid_angle_weights = weights * dw_factor
    gaunt_tensor = torch.einsum('si,sj,sk,s->ijk', A, A, A, solid_angle_weights)
    
    return directions, sqrt_weights, A, A_weighted, AT_A_weighted, AT_A, gaunt_tensor

def sample_env_map_torch(env_map, directions):
    H, W = env_map.shape[:2]
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    theta = torch.acos(z.clamp(-1, 1))
    phi = torch.where(torch.atan2(y, x) < 0, torch.atan2(y, x) + 2 * torch.pi, torch.atan2(y, x))
    u = (phi / (2 * torch.pi) * W).long() % W
    v = (theta / torch.pi * H).long() % H
    return env_map[v, u, :]

def compute_global_sh_coeffs(env_map, directions, sqrt_weights, A_weighted, AT_A_weighted):
    colors_weighted = sample_env_map_torch(env_map, directions) * sqrt_weights.view(-1, 1)
    return torch.linalg.solve(AT_A_weighted, A_weighted.T @ colors_weighted)

def srgb_to_linear_torch(c_srgb): return torch.where(c_srgb <= 0.04045, c_srgb / 12.92, torch.pow((c_srgb + 0.055) / 1.055, 2.4))
def linear_to_srgb_torch(c_linear): return torch.where(c_linear <= 0.0031308, c_linear * 12.92, 1.055 * torch.pow(c_linear.clamp(min=1e-10), 1.0 / 2.4) - 0.055)

@torch.no_grad()
def compute_brdf_attenuation_al(features_rest, L_source, l_max):
    """
    Computes A_l = ||B_lm|| / ||L_lm|| per band.
    This extracts the exact BRDF roughness profile and specular albedo from the source lighting.
    """
    N = features_rest.shape[0]
    A_l = torch.zeros(N, l_max, 3, device=features_rest.device)
    eps = 1e-5
    
    for l in range(1, l_max + 1):
        # Object rest bands (indices l^2 - 1 to (l+1)^2 - 1 in features_rest)
        obj_start = l**2 - 1
        obj_end = (l+1)**2 - 1
        E_obj = features_rest[:, obj_start:obj_end, :].norm(dim=1)  # [N, 3]

        # Source lighting bands (indices l^2 to (l+1)^2 in global L)
        l_start = l**2
        l_end = (l+1)**2
        E_src = L_source[l_start:l_end, :].norm(dim=0).unsqueeze(0)  # [1, 3]

        # A_l acts as the frequency-dependent attenuation (roughness + intensity)
        A_l[:, l-1, :] = E_obj / (E_src + eps)
        
    return A_l

@torch.no_grad()
def apply_al_specular_transfer_gpu(A_l, normals, sh_offset, A, AT_A_inv, directions, L_shad_target, l_max, device, batch_size=5000):
    """
    Evaluates target lighting at reflected directions and applies the data-driven BRDF attenuation A_l.
    """
    N = normals.shape[0]
    num_coeffs = (l_max + 1)**2
    adjusted_rest = torch.zeros(N, num_coeffs - 1, 3, device=device, dtype=torch.float32)
    M = torch.matmul(AT_A_inv, A.T)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        n_batch = normals[start:end]
        L_tgt_batch = L_shad_target[start:end]

        # 1. Mirror reflection of sampling directions about normal
        n_dot_d = torch.matmul(n_batch, directions.T)
        omega_r = directions.unsqueeze(0) - 2.0 * n_dot_d.unsqueeze(-1) * n_batch.unsqueeze(1)
        omega_r_sh = omega_r.clone()
        omega_r_sh[..., 0] = -omega_r_sh[..., 0]  # x-flip matches normal convention

        # 2. Evaluate Target lighting at pure reflected directions
        Y_L = evaluate_sh_bases_torch(omega_r_sh.reshape(-1, 3), l_max, device).reshape(B, directions.shape[0], num_coeffs)
        L_tgt_eval = torch.bmm(Y_L, L_tgt_batch).clamp(min=0.0)

        # 3. Project evaluated pure reflection back into SH
        L_tgt_srgb = linear_to_srgb_torch(L_tgt_eval)
        L_prime = torch.matmul(M.unsqueeze(0), L_tgt_srgb) - sh_offset

        # 4. Apply data-driven BRDF Attenuation (A_l) instead of hardcoded Gaussian blur
        for l in range(1, l_max + 1):
            adjusted_rest[start:end, l**2-1:(l+1)**2-1, :] = L_prime[:, l**2:(l+1)**2, :] * A_l[start:end, l-1, :].unsqueeze(1)

    return adjusted_rest


# ============================================================================
# PHASE 1: OFFLINE BAKING (VISIBILITY TO SH)
# ============================================================================
def phase_1_bake_visibility_sh(gaussians, directions, A_weighted, AT_A_weighted, sqrt_weights, device, res=512):
    """
    Renders visibility via 2DGS rasterizer and directly integrates it into Spherical Harmonics (V_lm).
    Optimized with Hemisphere and Front-Face Culling.
    Uses Exponential Shadow Maps (ESM) + 8-tap Poisson PCF for smooth visibility.
    """
    try:
        from gaussian_renderer import render
    except ImportError:
        raise ImportError("Could not import 'render' from 'gaussian_renderer'. Run from 2DGS root.")

    N, S, K = gaussians['xyz'].shape[0], directions.shape[0], A_weighted.shape[1]
    xyz = gaussians['xyz'].detach()
    
    scene_center = xyz.mean(dim=0)
    scene_radius = torch.max(torch.norm(xyz - scene_center, dim=1)).item() * 1.8

    mock_gaussians = MockGaussianModel(gaussians)
    pipe = MockPipeline()
    bg_color = torch.zeros(3, dtype=torch.float32, device=device)
    visibility_color = torch.ones((N, 3), dtype=torch.float32, device=device)
    max_scales, _ = torch.max(mock_gaussians.get_scaling.detach(), dim=1)
    per_gaussian_bias = max_scales * 3.0

    esm_alpha = 8.0 / scene_radius

    poisson_disk = torch.tensor([
        [-0.94201624, -0.39906216], [ 0.94558609, -0.76890725],
        [-0.09418410, -0.92938870], [ 0.34495938,  0.29387760],
        [-0.91588581,  0.45771432], [-0.81544232, -0.87912464],
        [-0.38277543,  0.27676845], [ 0.97484398,  0.75648379],
    ], device=device, dtype=torch.float32)
    filter_radius = 1.5

    AT_b = torch.zeros((K, N), dtype=torch.float32, device=device)

    fov_const = 2.0 * math.asin(1.0 / 5.0) * 1.2
    f_const = 1.0 / math.tan(fov_const / 2.0)
    half_res = (res - 1) * 0.5

    for i in tqdm(range(S), desc="Phase 1: Baking Shadow Maps"):
        # ALIGNMENT FIX: SH sampling directions have a flipped X relative to world/physical space.
        # Flip X so the camera is placed at the physically correct world-space direction,
        # making V_lm consistent with L_lm (both in SH space) for the Gaunt diffuse transfer.
        # Gen by Cursor
        dir_vec = directions[i].clone()
        dir_vec[0] = -dir_vec[0]

        cam, W2C, _ = create_telephoto_camera(scene_center, dir_vec, scene_radius, res, res, device)

        with torch.no_grad():
            render_pkg = render(cam, mock_gaussians, pipe, bg_color, override_color=visibility_color)
            if "surf_depth" in render_pkg:
                depth_map = render_pkg["surf_depth"][0]
            elif "depth" in render_pkg:
                depth_map = render_pkg["depth"][0]
            else:
                raise ValueError("Rasterizer didn't return depth (expected 'surf_depth' or 'depth')")

        xyz_view = torch.matmul(xyz, W2C[:3, :3]) + W2C[3, :3]
        depth_gaussian = xyz_view[:, 2]

        ndc_x = (xyz_view[:, 0] * f_const) / depth_gaussian
        ndc_y = (xyz_view[:, 1] * f_const) / depth_gaussian
        base_u = (ndc_x + 1.0) * half_res + 0.5
        base_v = (ndc_y + 1.0) * half_res + 0.5

        pcf_vis = torch.zeros(N, device=device)
        for off in poisson_disk:
            u_idx = torch.clamp((base_u + off[0] * filter_radius).long(), 0, res - 1)
            v_idx = torch.clamp((base_v + off[1] * filter_radius).long(), 0, res - 1)
            sd = depth_map[v_idx, u_idx]
            valid = (sd > 0).float()
            penetration = (depth_gaussian - sd - per_gaussian_bias).clamp(min=0.0)
            esm_vis = torch.exp(-esm_alpha * penetration)
            pcf_vis += valid * esm_vis + (1.0 - valid)
        pcf_vis /= len(poisson_disk)

        v_s_weighted = pcf_vis * sqrt_weights[i]
        AT_b += torch.outer(A_weighted[i], v_s_weighted)
                
    V_lm = torch.linalg.solve(AT_A_weighted, AT_b).T              
    return V_lm


# ============================================================================
# PHASE 2: DECOUPLED RELIGHTING
# ============================================================================
@torch.no_grad()
def phase_2_decoupled_relight(object_gaussians, all_normals, V_lm, L_lm_source, L_lm_target, sh_coeffs_offset, A, AT_A_inv, gaunt_tensor, directions, l_max, device, floor_alpha=0.05, tau_max=3.0, specular_boost=1.0):
    """
    Performs real-time relighting utilizing decoupled Diffuse and Specular paths.
    - Diffuse (DC): Modulated via Gaunt Tensor for self-shadows.
    - Specular (Rest): SH Convolution Theorem extracting exact data-driven BRDF.
    """
    features_dc = object_gaussians['features_dc']
    features_rest = object_gaussians['features_rest']

    # ====================================================================
    # 1. GENERATE SHADOWED ENVIRONMENTS & DIFFUSE TRANSFER
    # ====================================================================
    print("  -> Applying Shadowed Diffuse Transfer (Gaunt Tensor on DC)...")
    L_shad_source = torch.einsum('ijk,ic,nj->nkc', gaunt_tensor, L_lm_source, V_lm) 
    L_shad_target = torch.einsum('ijk,ic,nj->nkc', gaunt_tensor, L_lm_target, V_lm)

    A_l_diff = torch.tensor([np.pi, 2*np.pi/3, 2*np.pi/3, 2*np.pi/3, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4, 0., 0., 0., 0., 0., 0., 0.], device=device, dtype=torch.float32)
    normals_for_sh = all_normals.clone()
    normals_for_sh[:, 0] = -normals_for_sh[:, 0]
    
    Y_hat = evaluate_sh_bases_torch(normals_for_sh, l_max, device) * A_l_diff.view(1, -1)
    E_source_clean = torch.clamp(torch.einsum('nk,nkc->nc', Y_hat, L_shad_source), min=0.0)
    E_target_clean = torch.clamp(torch.einsum('nk,nkc->nc', Y_hat, L_shad_target), min=0.0)

    E_unocc = torch.clamp(torch.einsum('nk,kc->nc', Y_hat, L_lm_source), min=0.0)
    floor = floor_alpha * E_unocc + 1e-5
    tau_diffuse = torch.clamp(E_target_clean / (E_source_clean + floor), min=0.0, max=tau_max)

    dc_vals = features_dc[:, 0, :]
    dc_srgb = (C0 * dc_vals + 0.5).clamp(min=0.0)
    dc_linear = srgb_to_linear_torch(dc_srgb)
    dc_new_linear = (dc_linear * tau_diffuse).clamp(min=0.0)
    dc_new_srgb = linear_to_srgb_torch(dc_new_linear)
    new_features_dc = ((dc_new_srgb - 0.5) / C0).unsqueeze(1)

    # ====================================================================
    # 2. SH CONVOLUTION SPECULAR TRANSFER 
    # ====================================================================
    print("  -> Computing Per-Band Deconvolution Transfer (l>=1 all specular)...")

    # Extract effective BRDF attenuation A_l from the source environment
    A_l_spec = compute_brdf_attenuation_al(features_rest, L_lm_source, l_max)

    # Apply A_l to the physically shifted target reflections for all rest bands (l=1,2,3)
    rest_reflect = apply_al_specular_transfer_gpu(
        A_l=A_l_spec, normals=all_normals, sh_offset=sh_coeffs_offset,
        A=A, AT_A_inv=AT_A_inv, directions=directions,
        L_shad_target=L_shad_target, l_max=l_max, device=device
    )

    # All rest bands (l >= 1) receive specular transfer (matches princeton)
    # Gen by Cursor: apply specular boost to all L>=1 bands
    new_features_rest = rest_reflect * specular_boost
    print(f"  -> Specular boost applied: {specular_boost}x")

    object_gaussians['features_dc'] = new_features_dc
    object_gaussians['features_rest'] = new_features_rest

# ============================================================================
# Main Execution Flow
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Radiance transfer with self-shadowing (SH Convolution)")
    parser.add_argument("--source", default=ENV_MAP_SOURCE_PATH, help="Source HDR env map path")
    parser.add_argument("--target", default=ENV_MAP_TARGET_PATH, help="Target HDR env map path")
    parser.add_argument("--ply",    default=OBJECT_PATH,         help="Input Gaussian PLY path")
    parser.add_argument("--output", default=RELIGHT_OUTPUT_PATH, help="Output relit PLY path")
    parser.add_argument("--floor_alpha", type=float, default=0.05, help="Fraction of unoccluded irradiance used as diffuse denominator floor")
    parser.add_argument("--tau_max",     type=float, default=3.0,  help="Hard clamp on diffuse transfer ratio tau_diffuse")
    parser.add_argument("--specular_boost", type=float, default=1.0, help="Scale factor applied to all L>=1 SH rest bands after specular transfer (default: 2.0)")  # Gen by Cursor
    args = parser.parse_args()

    ENV_MAP_SOURCE_PATH  = args.source
    ENV_MAP_TARGET_PATH  = args.target
    OBJECT_PATH          = args.ply
    RELIGHT_OUTPUT_PATH  = args.output

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_samples = 1200
    
    print("Loading Gaussian PLY and setting up constants...")
    print(f"  Source env : {ENV_MAP_SOURCE_PATH}")
    print(f"  Target env : {ENV_MAP_TARGET_PATH}")
    print(f"  Input PLY  : {OBJECT_PATH}")
    print(f"  Output PLY : {RELIGHT_OUTPUT_PATH}")
    object_gaussians = load_ply(OBJECT_PATH)
    l_max = object_gaussians['max_sh_degree']
    
    directions, sqrt_weights, A, A_weighted, AT_A_weighted, AT_A, gaunt_tensor = precompute_sh_sampling(num_samples, l_max, device)
    AT_A_inv = torch.linalg.inv(AT_A)
    
    if torch.all(object_gaussians['normal'] == 0):
        with torch.no_grad():
            object_gaussians['normal'] = quaternion2rotmat(F.normalize(object_gaussians['rotation'], dim=1))[..., 2]
    all_normals = F.normalize(object_gaussians['normal'], dim=1)

    print("\n" + "="*70)
    print("PHASE 1: OFFLINE BAKING (Generating V_lm Matrix)")
    print("="*70)
    torch.cuda.reset_peak_memory_stats(device)
    bake_start_time = time.time()
    
    with torch.no_grad():
        V_lm = phase_1_bake_visibility_sh(
            object_gaussians, directions, A_weighted, AT_A_weighted, sqrt_weights, device, res=1024
        )
        
    bake_end_time = time.time()
    print(f"-> Phase 1 Completed in {bake_end_time - bake_start_time:.2f} seconds")

    print("\n" + "="*70)
    print("PHASE 2: DECOUPLED RELIGHTING (Diffuse Shadowed + Specular SH Convolution)")
    print("="*70)
    relight_start_time = time.time()
    
    env_map_source = cv2.cvtColor(cv2.imread(ENV_MAP_SOURCE_PATH, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    env_map_target = cv2.cvtColor(cv2.imread(ENV_MAP_TARGET_PATH, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if env_map_target.shape[:2] != env_map_source.shape[:2]:
        env_map_target = cv2.resize(env_map_target, (env_map_source.shape[1], env_map_source.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    env_map_source_gpu = torch.from_numpy(env_map_source.astype(np.float32)).to(device)
    env_map_target_gpu = torch.from_numpy(env_map_target.astype(np.float32)).to(device)
    offset_gpu = torch.full_like(env_map_target_gpu, 0.5)

    with torch.no_grad():
        L_lm_source = compute_global_sh_coeffs(env_map_source_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)
        L_lm_target = compute_global_sh_coeffs(env_map_target_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)
        sh_coeffs_offset = compute_global_sh_coeffs(offset_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)

        # Gen by Cursor: Informational diagnostic — L_lm[0] is the DC SH coefficient, proportional
        # to mean radiance. The ratio below is the physically correct brightness change the relight
        # will apply. If target is dimmer than source, the output will be darker — this is correct.
        energy_source = L_lm_source[0].mean().item()
        energy_target = L_lm_target[0].mean().item()
        ratio = energy_target / (energy_source + 1e-8)
        print(f"  [HDR Energy] Source L00: {energy_source:.5f}  |  Target L00: {energy_target:.5f}  |  Ratio: {ratio:.4f} ({'brighter' if ratio > 1 else 'darker'} target)")
        print(f"  [HDR Energy] This ratio is physically correct — no manual scaling applied.\n")

        phase_2_decoupled_relight(
            object_gaussians, all_normals, V_lm,
            L_lm_source, L_lm_target, sh_coeffs_offset,
            A, AT_A_inv, gaunt_tensor, directions, l_max, device,
            floor_alpha=args.floor_alpha, tau_max=args.tau_max,
            specular_boost=args.specular_boost  # Gen by Cursor
        )
        
    relight_end_time = time.time()
    print(f"-> Phase 2 Completed in {relight_end_time - relight_start_time:.4f} seconds")
    
    print("\n" + "="*70)
    print(f"Saving PLY to {RELIGHT_OUTPUT_PATH}...")
    save_ply_from_dict(object_gaussians, RELIGHT_OUTPUT_PATH)
    print(f"Total Script Time: {time.time() - bake_start_time:.2f} seconds")
    print(f"GPU Peak memory: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
