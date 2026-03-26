
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
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# ============================================================================
# SH Constants (unchanged)
# ============================================================================
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

# ============================================================================
# Configuration: File Paths
# ============================================================================
ENV_MAP_SOURCE_PATH = '/scratch/shared/by12/Transplat/envmap/city.hdr'
ENV_MAP_TARGET_PATH = '/scratch/shared/by12/Transplat/envmap/CombinedPano/garage/garage_enhance80.hdr'
OBJECT_PATH = '/scratch/shared/by12/Transplat/code/2d-gaussian-splatting/output/lego/point_cloud/iteration_30000/vanilla_point_cloud.ply'
RELIGHT_OUTPUT_PATH = "/scratch/shared/by12/Transplat/code/2d-gaussian-splatting/output/lego/point_cloud/iteration_30000/vanilla_city_to_garage_point_cloud.ply"

# Note: We no longer need SH_COEFFS_OUTPUT_PATH for the massive per-normal tensor

# ============================================================================
# PLY I/O Functions (Adapted for 2D GS)
# ============================================================================
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


def rotate_env_map(env_map, rotation_matrix):
    """UNCHANGED from original - only called once"""
    height, width, _ = env_map.shape

    lon = np.linspace(0, 2 * np.pi, width, endpoint=False)
    lat = np.linspace(0, np.pi, height)
    lon, lat = np.meshgrid(lon, lat)

    x = np.sin(lat) * np.cos(lon)
    y = np.sin(lat) * np.sin(lon)
    z = np.cos(lat)
    coords = np.stack([-x, y, z], axis=-1)

    H, W, _ = env_map.shape
    rotated_coords = (rotation_matrix.T @ coords.reshape(-1, 3).T).T.reshape(H, W, 3)
    lon_rot = np.arctan2(rotated_coords[..., 1], rotated_coords[..., 0])
    lat_rot = np.arccos(rotated_coords[..., 2]) 

    lon_rot[lon_rot < 0] += 2 * np.pi

    u = (lon_rot / (2 * np.pi) * width).astype(np.float32)
    v = (lat_rot / np.pi * height).astype(np.float32)

    u = np.clip(u, 0, width - 1)
    v = np.clip(v, 0, height - 1)
    rotated_map = cv2.remap(env_map, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    return rotated_map


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    num_samples = 1000
    rotation_matrix = np.eye(3)
    
    # Load Gaussians (Needs to be done before Precomputing SH sampling, because max_sh_degree is dynamically extracted now)
    print("Loading Gaussian PLY...")
    object_gaussians = load_ply(OBJECT_PATH)
    object_pcd = read_ply_to_point_cloud(OBJECT_PATH)
    l_max = object_gaussians['max_sh_degree']
    
    # Precompute SH sampling on GPU (done once)
    print("Precomputing SH basis on GPU...")
    directions, sqrt_weights, A, A_weighted, AT_A_weighted, AT_A = precompute_sh_sampling(num_samples, l_max, device)
    AT_A_inv = torch.linalg.inv(AT_A)  # Gen by Cursor - precompute inverse for fast matmul (16x16, negligible cost)
    
    # Load environment maps
    print("Loading environment maps...")
    env_map_source = cv2.imread(ENV_MAP_SOURCE_PATH, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    env_map_target = cv2.imread(ENV_MAP_TARGET_PATH, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    
    if env_map_source is None:
        raise IOError(f"Error loading source environment map: {ENV_MAP_SOURCE_PATH}")
    if env_map_target is None:
        raise IOError(f"Error loading target environment map: {ENV_MAP_TARGET_PATH}")
    
    env_map_source = cv2.cvtColor(env_map_source, cv2.COLOR_BGR2RGB)
    env_map_target = cv2.cvtColor(env_map_target, cv2.COLOR_BGR2RGB)
    if env_map_target.shape[:2] != env_map_source.shape[:2]:
        env_map_target = cv2.resize(env_map_target, (env_map_source.shape[1], env_map_source.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    # Handle zero normals
    if torch.all(object_gaussians['normal'] == 0):
        print("All normals are zero, converting from quaternions...")
        with torch.no_grad():
            normalized_quat = F.normalize(object_gaussians['rotation'], dim=1)
            rot_matrices = quaternion2rotmat(normalized_quat)
            new_normals = rot_matrices[..., 2]
            object_gaussians['normal'] = new_normals


    original_object_normals = object_gaussians['normal']  # Keep on GPU
    
    # Rotate normals and environment map
    all_normals = rotate_normals_torch(original_object_normals, rotation_matrix, device)
    env_map_source = rotate_env_map(env_map_source, rotation_matrix)
    
    # Normalize normals
    norm = all_normals.norm(dim=1, keepdim=True)
    all_normals = all_normals / norm
    
    # Convert env maps to GPU tensors
    H, W = env_map_target.shape[:2]
    env_map_source_gpu = torch.from_numpy(env_map_source.astype(np.float32)).to(device)
    env_map_target_gpu = torch.from_numpy(env_map_target.astype(np.float32)).to(device)
    
    # Offset for numerical stability (same as original)
    offset_np = env_map_target.astype(np.float32) * 0 + 0.5
    offset_gpu = torch.from_numpy(offset_np).to(device)
    
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
        # Step 1: Compute GLOBAL SH Coefficients for Environment Maps
        # ========================================================================
        print("Computing GLOBAL SH coefficients for environment maps...")
        
        # Compute source, target, and offset SH coeffs (Raw, unshaded)
        sh_coeffs_source_global = compute_global_sh_coeffs(env_map_source_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)
        sh_coeffs_target_global = compute_global_sh_coeffs(env_map_target_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)
        sh_coeffs_offset = compute_global_sh_coeffs(offset_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)
        
        print(f"  Global SH source mean DC: {sh_coeffs_source_global[0].mean():.4f}")
        print(f"  Global SH target mean DC: {sh_coeffs_target_global[0].mean():.4f}")

        # ========================================================================
        # Step 2: Define Lambertian Kernel Coefficients (A_l)
        # ========================================================================
        # These coefficients convert Radiance -> Irradiance via convolution
        A_l = torch.tensor([
            np.pi,                              # l=0 (1 coeff)
            2*np.pi/3, 2*np.pi/3, 2*np.pi/3,    # l=1 (3 coeffs)
            np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4,  # l=2 (5 coeffs)
            0., 0., 0., 0., 0., 0., 0.          # l=3 (7 coeffs, all zero)
        ], device=device, dtype=torch.float32)  # [16]
        
        print(f"  Lambertian kernel A_l applied.")

        # ========================================================================
        # Step 3: Analytic Irradiance Computation (E = Y_lm * (A_l * L_lm))
        # ========================================================================
        print("Computing irradiance E(n) for all normals (Analytic)...")
        
        # FIX: Align X-axis - negate x component to match the coordinate convention
        # used in env map projection. This ensures lighting doesn't mirror horizontally.
        normals_for_sh = all_normals.clone()
        normals_for_sh[:, 0] = -normals_for_sh[:, 0]
        
        # Evaluate Y_lm for all normals (Efficient vectorization)
        Y_lm_normals = evaluate_sh_bases_torch(normals_for_sh, l_max, device)  # [N, 16]
        
        # Convolve: L_hat_lm = A_l * L_lm
        irradiance_source_coeffs = sh_coeffs_source_global * A_l.view(-1, 1)  # [16, 3]
        irradiance_target_coeffs = sh_coeffs_target_global * A_l.view(-1, 1)  # [16, 3]
        
        # Compute E(n) = Σ Y_lm(n) * L_hat_lm
        E_source = Y_lm_normals @ irradiance_source_coeffs  # [N, 3]
        E_target = Y_lm_normals @ irradiance_target_coeffs  # [N, 3]
        
        print(f"  E_source range: [{E_source.min():.4f}, {E_source.max():.4f}]")
        print(f"  E_target range: [{E_target.min():.4f}, {E_target.max():.4f}]")

        # ========================================================================
        # Step 4: Calculate Scalar Transfer Ratio (Irradiance Ratio)
        # ========================================================================
        # T_diffuse = E_target(n) / E_source(n)
        epsilon = 1e-6
        T_diffuse = E_target / (E_source.abs() + epsilon)  # [N, 3]
        T_diffuse = torch.clamp(T_diffuse, min=-5.0, max=5.0)  # Prevent extreme values
        
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
        )
        
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




