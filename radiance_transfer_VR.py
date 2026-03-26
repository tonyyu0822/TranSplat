import torch
import numpy as np
from torch import nn
from plyfile import PlyData, PlyElement
import open3d as o3d
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import os
import time
import math
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
# Gen by Cursor - Use pre-sampled probe SH coefficient txt files directly
# Format: C0..C8 per channel (R, G, B) = degree-2 SH (9 coeffs each)
PROBE_SH_SOURCE_PATH = '/scratch/shared/by12/Transplat/code/2d-gaussian-splatting_SG/VR/bunny_px_02-23_city.txt'  # source lighting (city)
PROBE_SH_TARGET_PATH = '/scratch/shared/by12/Transplat/code/2d-gaussian-splatting_SG/VR/bunny_px_02-23_px.txt'    # target lighting (px)
OBJECT_PATH = '/scratch/shared/by12/Transplat/code/2d-gaussian-splatting/output/VR/bunny_city/point_cloud/iteration_30000/point_cloud.ply'
RELIGHT_OUTPUT_PATH = "/scratch/shared/by12/Transplat/code/2d-gaussian-splatting/output/VR/bunny_city/point_cloud/iteration_30000/bunny_city_px_point_cloud.ply"

def load_ply(path, normal_b=True):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    if normal_b:
        normal = np.stack(
            (
                np.asarray(plydata.elements[0]["nx"]),
                np.asarray(plydata.elements[0]["ny"]),
                np.asarray(plydata.elements[0]["nz"]),
            ),
            axis=1,
        )
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    num_rest_features = len(extra_f_names) // 3
    import math
    max_sh_degree = int(math.sqrt(num_rest_features + 1)) - 1
    print(f"Detected SH degree: {max_sh_degree}")
    
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    _xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    _features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    _features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    _opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    _scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    _rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    if normal_b:
        _normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))

    gaussians = {
        "xyz": _xyz,
        "features_dc": _features_dc,
        "features_rest": _features_rest,
        "opacity": _opacity,
        "scaling": _scaling,
        "rotation": _rotation,
        "normal": _normal,
        "max_sh_degree": max_sh_degree
    }
    return gaussians


def save_ply_from_dict(dict, save_path):
    def construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(_features_dc.shape[1]*_features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(_features_rest.shape[1]*_features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(_rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    xyz = dict['xyz'].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = dict['features_dc'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = dict['features_rest'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = dict['opacity'].detach().cpu().numpy()
    scale = dict['scaling'].detach().cpu().numpy()
    rotation = dict['rotation'].detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(dict['features_dc'], dict['features_rest'],  dict['scaling'], dict['rotation'])]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_path)


def read_ply_to_point_cloud(ply_path):
    plydata = PlyData.read(ply_path)
    xyz = np.vstack([plydata['vertex'][att] for att in ['x', 'y', 'z']]).T
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))


def quaternion2rotmat(q):
    r, x, y, z = q.split(1, -1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)
        ], -1).reshape([len(q), 3, 3])
    return R


# ============================================================================
# GPU-Optimized SH Functions
# ============================================================================
def evaluate_sh_bases_torch(directions, l_max, device):
    """
    Evaluate SH bases for batch of directions on GPU.
    """
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    
    bases = [torch.full((directions.shape[0],), C0, device=device, dtype=directions.dtype)]
    
    if l_max > 0:
        bases.extend([
            -C1 * y,
            C1 * z,
            -C1 * x
        ])
        
        if l_max > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            bases.extend([
                C2[0] * xy,
                C2[1] * yz,
                C2[2] * (2.0 * zz - xx - yy),
                C2[3] * xz,
                C2[4] * (xx - yy)
            ])
            
            if l_max > 2:
                bases.extend([
                    C3[0] * y * (3 * xx - yy),
                    C3[1] * xy * z,
                    C3[2] * y * (4 * zz - xx - yy),
                    C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                    C3[4] * x * (4 * zz - xx - yy),
                    C3[5] * z * (xx - yy),
                    C3[6] * x * (xx - 3 * yy)
                ])
                
                if l_max > 3:
                    bases.extend([
                        C4[0] * xy * (xx - yy),
                        C4[1] * yz * (3 * xx - yy),
                        C4[2] * xy * (7 * zz - 1),
                        C4[3] * yz * (7 * zz - 3),
                        C4[4] * (zz * (35 * zz - 30) + 3),
                        C4[5] * xz * (7 * zz - 3),
                        C4[6] * (xx - yy) * (7 * zz - 1),
                        C4[7] * xz * (xx - 3 * yy),
                        C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                    ])
    
    return torch.stack(bases, dim=1)  # [N, num_coeffs]


def precompute_sh_sampling(num_samples, l_max, device):
    """
    Precompute all sampling-related tensors on GPU.
    """
    indices = torch.arange(num_samples, device=device, dtype=torch.float64)
    theta = torch.acos(1 - 2 * (indices + 0.5) / num_samples)
    phi = torch.pi * (1 + 5 ** 0.5) * indices
    
    directions = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], dim=1).float()  # [num_samples, 3]
    
    weights = torch.sin(theta).float()
    sqrt_weights = torch.sqrt(weights)
    
    # SH basis matrix
    A = evaluate_sh_bases_torch(directions, l_max, device)  # [num_samples, num_coeffs]
    A_weighted = A * sqrt_weights.unsqueeze(1)
    
    # Precompute for weighted least squares (used in environment_map_to_sh_coefficients)
    AT_A_weighted = A_weighted.T @ A_weighted  # [num_coeffs, num_coeffs]
    
    # Precompute for unweighted least squares (used in adjust_sh_coeffs)
    AT_A = A.T @ A  # [num_coeffs, num_coeffs]
    
    return directions, sqrt_weights, A, A_weighted, AT_A_weighted, AT_A


def sample_env_map_torch(env_map, directions):
    """
    Sample environment map for batch of directions on GPU.
    """
    H, W = env_map.shape[:2]
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    
    # Convert directions to spherical coordinates (identical to original)
    theta = torch.acos(z.clamp(-1, 1))
    phi = torch.atan2(y, x)
    phi = torch.where(phi < 0, phi + 2 * torch.pi, phi)
    
    # Map to pixel coordinates (identical to original)
    u = (phi / (2 * torch.pi) * W).long() % W
    v = (theta / torch.pi * H).long() % H
    
    # Sample using advanced indexing
    colors = env_map[v, u, :]
    return colors


def compute_global_sh_coeffs(env_map, directions, sqrt_weights, A_weighted, AT_A_weighted):
    """
    Compute GLOBAL SH coefficients for an environment map.
    Does NOT apply cosine convolution (n.w). This is the raw lighting projection.
    """
    # Sample colors from env map at sample directions (no shading)
    colors = sample_env_map_torch(env_map, directions)  # [num_samples, 3]
    
    # Apply weights
    colors_weighted = colors * sqrt_weights.view(-1, 1)
    
    # Weighted least squares solve
    AT_b = A_weighted.T @ colors_weighted
    sh_coeffs = torch.linalg.solve(AT_A_weighted, AT_b)  # [num_coeffs, 3]
    
    return sh_coeffs


def srgb_to_linear_torch(c_srgb):
    """IDENTICAL to srgb_to_linear()"""
    threshold = 0.04045
    return torch.where(
        c_srgb <= threshold,
        c_srgb / 12.92,
        torch.pow((c_srgb + 0.055) / 1.055, 2.4)
    )


def linear_to_srgb_torch(c_linear):
    """IDENTICAL to linear_to_srgb()"""
    threshold = 0.0031308
    return torch.where(
        c_linear <= threshold,
        c_linear * 12.92,
        1.055 * torch.pow(c_linear.clamp(min=1e-10), 1.0 / 2.4) - 0.055
    )


def adjust_sh_coeffs_gpu(sh_coeffs, A, AT_A, to_linear=True, batch_size=10000):
    """
    GPU-batched version of adjust_sh_coeffs().
    """
    N = sh_coeffs.shape[0]
    num_samples, num_coeffs = A.shape
    device = sh_coeffs.device
    
    adjusted = torch.zeros_like(sh_coeffs)
    
    for start in tqdm(range(0, N, batch_size), desc="Adjusting SH Coeffs (GPU)"):
        end = min(start + batch_size, N)
        batch = sh_coeffs[start:end]  # [B, num_coeffs, 3]
        B = batch.shape[0]
        
        # Evaluate colors at sample directions: colors[b, s, c] = sum_k A[s, k] * batch[b, k, c]
        colors = torch.einsum('sk,bkc->bsc', A, batch)  # [B, num_samples, 3]
        
        # Color space conversion
        if to_linear:
            colors = srgb_to_linear_torch(colors)
        else:
            colors = linear_to_srgb_torch(colors)
        
        # Solve least squares: solution = (A.T @ A)^-1 @ A.T @ colors
        # AT_b[b, k, c] = sum_s A[s, k] * colors[b, s, c]
        AT_b = torch.einsum('sk,bsc->bkc', A, colors)  # [B, num_coeffs, 3]
        
        # Batch solve: [B*3, num_coeffs]
        AT_b_flat = AT_b.permute(0, 2, 1).reshape(B * 3, num_coeffs)
        solution_flat = torch.linalg.solve(AT_A, AT_b_flat.T).T  # [B*3, num_coeffs]
        solution = solution_flat.reshape(B, 3, num_coeffs).permute(0, 2, 1)  # [B, num_coeffs, 3]
        
        adjusted[start:end] = solution
    
    return adjusted


@torch.no_grad()
def apply_transfer_fused_gpu(sh_coeffs, T_diffuse, sh_offset, A, AT_A_inv, batch_size=20000):
    """
    Gen by Cursor - Fused sRGB->linear + transfer + linear->sRGB in a single pass.
    Optimizations vs. two separate adjust_sh_coeffs_gpu calls:
      1. Halves the matmuls & solves (one pass instead of two)
      2. Uses precomputed (A^T A)^{-1} matmul instead of torch.linalg.solve
      3. Uses torch.matmul instead of einsum for lower dispatch overhead
      4. @torch.no_grad() avoids autograd overhead from nn.Parameter tensors
    
    Args:
        sh_coeffs: [N, num_coeffs, 3] original Gaussian SH coefficients (sRGB)
        T_diffuse: [N, 3] per-normal transfer ratio
        sh_offset: [num_coeffs, 3] offset for numerical stability
        A: [num_samples, num_coeffs] SH basis matrix
        AT_A_inv: [num_coeffs, num_coeffs] precomputed (A.T @ A)^{-1}
        batch_size: batch size for GPU memory management
    Returns:
        [N, num_coeffs, 3] transferred SH coefficients (sRGB)
    """
    N = sh_coeffs.shape[0]
    num_coeffs = A.shape[1]
    device = sh_coeffs.device
    AT = A.T  # [K, S] - precompute transpose
    
    adjusted = torch.zeros(N, num_coeffs, 3, device=device, dtype=sh_coeffs.dtype)
    
    for start in tqdm(range(0, N, batch_size), desc="Fused Transfer (GPU)"):
        end = min(start + batch_size, N)
        batch = sh_coeffs[start:end] + sh_offset  # [B, K, 3]
        batch_T = T_diffuse[start:end]  # [B, 3]
        
        # Evaluate colors at sample directions: [S, K] @ [B, K, 3] -> [B, S, 3]
        colors = torch.matmul(A, batch)
        
        # sRGB -> linear
        colors = srgb_to_linear_torch(colors)
        
        # Apply transfer ratio in color domain (equivalent to SH domain for uniform scaling)
        colors = colors * batch_T.unsqueeze(1)  # [B, S, 3]
        
        # linear -> sRGB
        colors = linear_to_srgb_torch(colors)
        
        # Project back to SH: [K, S] @ [B, S, 3] -> [B, K, 3]
        AT_b = torch.matmul(AT, colors)
        
        # Solve via precomputed inverse: [K, K] @ [B, K, 3] -> [B, K, 3]
        # AT_A is symmetric so AT_A_inv is also symmetric, no transpose needed
        solution = torch.matmul(AT_A_inv, AT_b)
        
        adjusted[start:end] = solution - sh_offset
    
    return adjusted


def rotate_normals_torch(normals, rotation_matrix, device):
    """GPU version of rotate_normals()"""
    rot = torch.from_numpy(rotation_matrix).float().to(device)
    return (rot @ normals.T).T


# Gen by Cursor - load_probe_sh_txt: parse pre-sampled probe SH coefficients
def load_probe_sh_txt(path, l_max, device):
    """
    Parse a probe SH coefficient txt file produced by a light probe sampler.
    Expected format per channel block:
        channel R
        C0: <value>
        C1: <value>
        ...
        C8: <value>
    Returns a [num_coeffs, 3] tensor on `device`, where num_coeffs = (l_max+1)^2.
    Coefficients from the file (degree-2, 9 values) are placed in the first 9 slots;
    higher-degree slots (l=3, indices 9-15) are zero-padded.
    """
    coeffs = {'R': [], 'G': [], 'B': []}
    current_channel = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('channel '):
                current_channel = line.split()[1]  # 'R', 'G', or 'B'
            elif line.startswith('C') and current_channel is not None:
                val = float(line.split(':')[1].strip())
                coeffs[current_channel].append(val)
    num_coeffs_total = (l_max + 1) ** 2
    out = torch.zeros(num_coeffs_total, 3, device=device, dtype=torch.float32)
    for ci, ch in enumerate(['R', 'G', 'B']):
        vals = coeffs[ch]  # 9 values (degree-2)
        n = min(len(vals), num_coeffs_total)
        out[:n, ci] = torch.tensor(vals[:n], dtype=torch.float32, device=device)
    return out  # [num_coeffs_total, 3]


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    num_samples = 1000
    rotation_matrix = np.eye(3)
    
    # Load Gaussians (needs to be done first to get max_sh_degree)
    print("Loading Gaussian PLY...")
    object_gaussians = load_ply(OBJECT_PATH)
    object_pcd = read_ply_to_point_cloud(OBJECT_PATH)
    l_max = object_gaussians['max_sh_degree']
    
    # Precompute SH sampling on GPU (done once)
    print("Precomputing SH basis on GPU...")
    directions, sqrt_weights, A, A_weighted, AT_A_weighted, AT_A = precompute_sh_sampling(num_samples, l_max, device)
    AT_A_inv = torch.linalg.inv(AT_A)  # Gen by Cursor - precompute inverse for fast matmul (16x16, negligible cost)
    
    # Gen by Cursor - Load pre-sampled probe SH coefficients directly from txt files
    # Skips env map loading and SH estimation entirely
    print("Loading probe SH coefficients from txt files...")
    print(f"  Source (city): {PROBE_SH_SOURCE_PATH}")
    print(f"  Target (px):   {PROBE_SH_TARGET_PATH}")
    sh_coeffs_source_global = load_probe_sh_txt(PROBE_SH_SOURCE_PATH, l_max, device)  # [num_coeffs, 3]
    sh_coeffs_target_global = load_probe_sh_txt(PROBE_SH_TARGET_PATH, l_max, device)  # [num_coeffs, 3]
    print(f"  Loaded {sh_coeffs_source_global.shape[0]} SH coefficients per channel (l_max={l_max})")
    
    # Handle zero normals
    if torch.all(object_gaussians['normal'] == 0):
        print("All normals are zero, converting from quaternions...")
        with torch.no_grad():
            normalized_quat = F.normalize(object_gaussians['rotation'], dim=1)
            rot_matrices = quaternion2rotmat(normalized_quat)
            new_normals = rot_matrices[..., 2]
            object_gaussians['normal'] = new_normals

    original_object_normals = object_gaussians['normal']  # Keep on GPU
    
    # Rotate normals (no env map rotation needed - probe SH already in probe space)
    all_normals = rotate_normals_torch(original_object_normals, rotation_matrix, device)
    
    # Normalize normals
    norm = all_normals.norm(dim=1, keepdim=True)
    all_normals = all_normals / norm
    
    # Gen by Cursor - Constant offset tensor for numerical stability (replaces env-image-based offset)
    num_coeffs_total = (l_max + 1) ** 2
    sh_coeffs_offset = torch.full((num_coeffs_total, 3), 0.5, device=device, dtype=torch.float32)
    
    # ========================================================================
    # START TIMING: Radiance Transfer Computation
    # ========================================================================
    print("\n" + "="*70)
    print("Starting radiance transfer computation...")
    print("="*70)
    # Gen by Cursor - Memory tracking
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    transfer_start_time = time.time()
    
    # Gen by Cursor - Disable autograd for entire computation block
    # The gaussian params have requires_grad=True but we don't need gradients here
    with torch.no_grad():
        # ========================================================================
        # Step 1: Use Pre-Sampled Probe SH Coefficients (loaded above)
        # ========================================================================
        # Gen by Cursor - sh_coeffs_source_global and sh_coeffs_target_global are
        # already loaded from txt files; no env map estimation needed.
        print("Using pre-loaded probe SH coefficients...")
        print(f"  SH source (city) DC mean: {sh_coeffs_source_global[0].mean():.4f}")
        print(f"  SH target (px)   DC mean: {sh_coeffs_target_global[0].mean():.4f}")

        # ========================================================================
        # Step 2: DC-only Transfer Ratio (Stable per-channel color shift)
        # ========================================================================
        # Gen by Cursor - WHY DC-only instead of full irradiance ratio:
        # The per-normal irradiance ratio E_target(n)/E_source(n) is numerically
        # unstable with probe SH data. The city probe has strong L1 terms that
        # almost perfectly CANCEL its DC for certain normal directions (e.g. upward
        # normals: E_source(B,up)≈0.029 nearly zero), while the px probe has near-zero
        # L1 terms so E_target stays close to its DC (≈0.154). The ratio becomes
        # 0.154/0.029 ≈ 5.3x → clamped to 5 → B channel blows out from top-down view.
        # This instability does NOT appear with HDR envmaps because the WLS-estimated
        # SH coefficients are smoother and L1 terms don't create near-cancellation.
        #
        # DC ratio = L_0^0_target / L_0^0_source gives a stable global color temperature
        # shift per channel. For city→px: T_R≈0.81, T_G≈T_B≈0.33 → correctly red/orange.
        epsilon = 1e-6
        dc_source = sh_coeffs_source_global[0, :]  # [3] - L_0^0 per channel (mean radiance)
        dc_target = sh_coeffs_target_global[0, :]  # [3]
        T_per_channel = (dc_target / dc_source.clamp(min=epsilon))  # [3]
        T_per_channel = torch.clamp(T_per_channel, min=0.0, max=5.0)
        N = all_normals.shape[0]
        T_diffuse = T_per_channel.unsqueeze(0).expand(N, -1).contiguous()  # [N, 3] uniform

        print(f"  DC transfer ratio T_R={T_per_channel[0]:.4f}, T_G={T_per_channel[1]:.4f}, T_B={T_per_channel[2]:.4f}")
        
        print(f"  T_diffuse range: [{T_diffuse.min():.4f}, {T_diffuse.max():.4f}], mean: {T_diffuse.mean():.4f}")

        # ========================================================================
        # Step 5: Apply Transfer to Gaussian SH Coefficients (Fused)
        # ========================================================================
        features_dc = object_gaussians['features_dc']  # [N, 1, 3]
        features_rest = object_gaussians['features_rest']  # [N, 15, 3]
        gaussians_sh_coeffs = torch.cat([features_dc, features_rest], dim=1)  # [N, num_coeffs, 3]

        # Gen by Cursor - Fused: sRGB->linear + transfer + linear->sRGB in single pass
        # Replaces two separate adjust_sh_coeffs_gpu calls with one fused pass
        print("Applying fused transfer (sRGB -> linear -> transfer -> sRGB)...")
        gaussians_sh_coeffs_adjusted = apply_transfer_fused_gpu(
            gaussians_sh_coeffs, T_diffuse, sh_coeffs_offset, A, AT_A_inv, batch_size=20000
        )  # sh_coeffs_offset is now a constant [num_coeffs, 3] tensor of 0.5
        
        # Split back
        features_dc_adjusted = gaussians_sh_coeffs_adjusted[:, 0:1, :]
        features_rest_adjusted = gaussians_sh_coeffs_adjusted[:, 1:, :]
        
        # Update gaussians
        object_gaussians['features_dc'] = features_dc_adjusted
        object_gaussians['features_rest'] = features_rest_adjusted
    
    # ========================================================================
    # END TIMING: Radiance Transfer Computation
    # ========================================================================
    transfer_end_time = time.time()
    transfer_elapsed = transfer_end_time - transfer_start_time
    # Gen by Cursor - Memory usage reporting
    mem_after = torch.cuda.memory_allocated(device)
    mem_peak = torch.cuda.max_memory_allocated(device)
    print("\n" + "="*70)
    print(f"Radiance transfer computation completed in {transfer_elapsed:.2f} seconds ({transfer_elapsed/60:.2f} minutes)")
    print(f"GPU Memory before computation: {mem_before / 1024**2:.2f} MB")
    print(f"GPU Memory after computation:  {mem_after / 1024**2:.2f} MB")
    print(f"GPU Peak memory during computation: {mem_peak / 1024**2:.2f} MB")
    print(f"GPU Memory increase: {(mem_after - mem_before) / 1024**2:.2f} MB")
    print("="*70 + "\n")
    
    # Save result
    print(f"Saving relit PLY to {RELIGHT_OUTPUT_PATH}...")
    save_ply_from_dict(object_gaussians, RELIGHT_OUTPUT_PATH)
    
    print("Done!")




