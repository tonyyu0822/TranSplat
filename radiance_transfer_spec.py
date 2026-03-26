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
ENV_MAP_SOURCE_PATH = '/scratch/shared/by12/Transplat/envmap/city.hdr'
ENV_MAP_TARGET_PATH = '/scratch/shared/by12/Transplat/envmap/fireplace.hdr'
OBJECT_PATH = '/scratch/shared/by12/Transplat/code/2d-gaussian-splatting/output/ficus/point_cloud/iteration_30000/filtered_point_cloud.ply'
RELIGHT_OUTPUT_PATH = "/scratch/shared/by12/Transplat/code/2d-gaussian-splatting/output/ficus/point_cloud/iteration_30000/specV1_fireplace_point_cloud.ply"

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
    
    A = evaluate_sh_bases_torch(directions, l_max, device)  # [num_samples, num_coeffs]
    A_weighted = A * sqrt_weights.unsqueeze(1)
    
    AT_A_weighted = A_weighted.T @ A_weighted  # [num_coeffs, num_coeffs]
    AT_A = A.T @ A  # [num_coeffs, num_coeffs]
    
    return directions, sqrt_weights, A, A_weighted, AT_A_weighted, AT_A

def sample_env_map_torch(env_map, directions):
    H, W = env_map.shape[:2]
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    
    theta = torch.acos(z.clamp(-1, 1))
    phi = torch.atan2(y, x)
    phi = torch.where(phi < 0, phi + 2 * torch.pi, phi)
    
    u = (phi / (2 * torch.pi) * W).long() % W
    v = (theta / torch.pi * H).long() % H
    
    colors = env_map[v, u, :]
    return colors

def compute_global_sh_coeffs(env_map, directions, sqrt_weights, A_weighted, AT_A_weighted):
    colors = sample_env_map_torch(env_map, directions)  # [num_samples, 3]
    colors_weighted = colors * sqrt_weights.view(-1, 1)
    
    AT_b = A_weighted.T @ colors_weighted
    sh_coeffs = torch.linalg.solve(AT_A_weighted, AT_b)  # [num_coeffs, 3]
    return sh_coeffs

def srgb_to_linear_torch(c_srgb):
    threshold = 0.04045
    return torch.where(
        c_srgb <= threshold,
        c_srgb / 12.92,
        torch.pow((c_srgb + 0.055) / 1.055, 2.4)
    )

def linear_to_srgb_torch(c_linear):
    threshold = 0.0031308
    return torch.where(
        c_linear <= threshold,
        c_linear * 12.92,
        1.055 * torch.pow(c_linear.clamp(min=1e-10), 1.0 / 2.4) - 0.055
    )

@torch.no_grad()
def apply_transfer_fused_gpu(sh_coeffs, T_diffuse, sh_offset, A, AT_A_inv, batch_size=20000):
    N = sh_coeffs.shape[0]
    num_coeffs = A.shape[1]
    device = sh_coeffs.device
    AT = A.T  
    
    adjusted = torch.zeros(N, num_coeffs, 3, device=device, dtype=sh_coeffs.dtype)
    
    for start in tqdm(range(0, N, batch_size), desc="Fused Diffuse Transfer"):
        end = min(start + batch_size, N)
        batch = sh_coeffs[start:end] + sh_offset  # [B, K, 3]
        batch_T = T_diffuse[start:end]  # [B, 3]
        
        colors = torch.matmul(A, batch)
        colors = srgb_to_linear_torch(colors)
        colors = colors * batch_T.unsqueeze(1)  # [B, S, 3]
        colors = linear_to_srgb_torch(colors)
        
        AT_b = torch.matmul(AT, colors)
        solution = torch.matmul(AT_A_inv, AT_b)
        
        adjusted[start:end] = solution - sh_offset
    return adjusted

@torch.no_grad()
def apply_specular_transfer_gpu(sh_coeffs, normals, sh_offset, A, AT_A_inv, directions, L_source_global, L_target_global, l_max, device, batch_size=5000):
    N, num_coeffs, _ = sh_coeffs.shape
    S = directions.shape[0]
    adjusted_rest = torch.zeros(N, num_coeffs - 1, 3, device=device, dtype=sh_coeffs.dtype)

    M = torch.matmul(AT_A_inv, A.T) # [16, S] Projection Matrix

    for start in tqdm(range(0, N, batch_size), desc="Analytic Specular Transfer"):
        end = min(start + batch_size, N)
        batch = sh_coeffs[start:end] + sh_offset        # [B, 16, 3]
        n_batch = normals[start:end]                    # [B, 3]

        n_dot_omega = torch.matmul(n_batch, directions.T) # [B, S]
        V = 2.0 * n_dot_omega.unsqueeze(-1) * n_batch.unsqueeze(1) - directions.unsqueeze(0) # [B, S, 3]

        Y_V = evaluate_sh_bases_torch(V.reshape(-1, 3), l_max, device).reshape(-1, S, num_coeffs) # [B, S, 16]
        colors_r = torch.bmm(Y_V, batch) # [B, S, 3]
        colors_r_linear = srgb_to_linear_torch(colors_r)

        F = torch.matmul(M.unsqueeze(0), colors_r_linear) # [B, 16, 3]

        F_prime = torch.zeros_like(F)
        for l in range(1, l_max + 1):
            start_idx = l**2
            end_idx = (l+1)**2
            
            E_F_l = torch.sqrt(torch.sum(F[:, start_idx:end_idx, :]**2, dim=1) + 1e-8) # [B, 3]
            E_L_l = torch.sqrt(torch.sum(L_source_global[start_idx:end_idx, :]**2, dim=0) + 1e-8) # [3]
            
            S_l = E_F_l / (E_L_l.unsqueeze(0) + 1e-6) # [B, 3]
            F_prime[:, start_idx:end_idx, :] = L_target_global[start_idx:end_idx, :].unsqueeze(0) * S_l.unsqueeze(1)

        colors_v_linear = torch.bmm(Y_V, F_prime) 
        colors_v_srgb = linear_to_srgb_torch(colors_v_linear)
        
        C_prime = torch.matmul(M.unsqueeze(0), colors_v_srgb) - sh_offset # [B, 16, 3]
        adjusted_rest[start:end] = C_prime[:, 1:, :]
        
    return adjusted_rest


# ============================================================================
# NEW: Geomerics / Hazel 2015 Non-Linear Diffuse Evaluation
# ============================================================================
def compute_nonlinear_irradiance(sh_coeffs, normals, A_l, device):
    """
    Evaluates diffuse irradiance using the Geomerics Non-Linear L1 Spherical Harmonics approximation.
    This strictly eliminates Gibbs ringing (black artifacts) and guarantees positivity 
    while perfectly preserving L0 energy and L1 directional moments for high-contrast lighting.
    """
    N = normals.shape[0]
    
    # Extract L0 (DC)
    L0 = sh_coeffs[0] # [3]
    
    # 1. Compute Ambient Irradiance (A)
    # A = Y_0 * A_l[0] * L0
    A = C0 * A_l[0] * L0 # [3]
    A = torch.clamp(A, min=1e-6) # prevent div by zero
    
    # 2. Extract Directional Irradiance Vector (D) from L1
    # SH Basis mapping: Y_{1,-1} = -C1*y, Y_{1,0} = C1*z, Y_{1,1} = -C1*x
    V_x = -C1 * sh_coeffs[3] * A_l[1] # [3]
    V_y = -C1 * sh_coeffs[1] * A_l[1] # [3]
    V_z =  C1 * sh_coeffs[2] * A_l[1] # [3]
    
    D_vec = torch.stack([V_x, V_y, V_z], dim=0) # [3, 3] (spatial, color)
    d_mag = torch.norm(D_vec, dim=0) # [3]
    D_dir = D_vec / (d_mag.unsqueeze(0) + 1e-8) # [3, 3]
    
    # 3. Evaluate standard directional dot product
    cos_theta = torch.matmul(normals, D_dir) # [N, 3]
    
    # 4. Geomerics Non-Linear L1 Fit
    E_out = torch.zeros(N, 3, device=device)
    
    for c in range(3):
        Ac = A[c]
        dc = d_mag[c]
        cos_c = cos_theta[:, c]
        
        if dc <= Ac:
            # Safe linear evaluation (no ringing possible)
            E_out[:, c] = Ac + dc * cos_c
        else:
            # Non-Linear lobe to prevent negative ringing
            # We calculate the optimal power 'p' to strictly preserve L0 and L1 energy
            r = (dc / (3.0 * Ac)).clamp(max=0.99)
            p = (2.0 * r) / (1.0 - r)
            
            # Evaluate energy-preserving squared lobe
            lobe = ((1.0 + cos_c) / 2.0) ** p
            E_out[:, c] = (p + 1.0) * Ac * lobe

    # 5. Add L2 band linearly to recover subtle high-frequency shadows
    if sh_coeffs.shape[0] >= 9:
        x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        
        # Standard L2 Basis Evaluation
        Y_L2 = torch.stack([
            C2[0] * xy,
            C2[1] * yz,
            C2[2] * (2.0 * zz - xx - yy),
            C2[3] * xz,
            C2[4] * (xx - yy)
        ], dim=1) # [N, 5]
        
        L2_coeffs = sh_coeffs[4:9] * A_l[4:9].unsqueeze(1) # [5, 3]
        E_L2 = torch.matmul(Y_L2, L2_coeffs) # [N, 3]
        
        # Add L2, soft-clamped to strictly guarantee the absolute positivity
        E_out = torch.clamp(E_out + E_L2, min=0.0)
    
    return E_out


def rotate_normals_torch(normals, rotation_matrix, device):
    rot = torch.from_numpy(rotation_matrix).float().to(device)
    return (rot @ normals.T).T

def rotate_env_map(env_map, rotation_matrix):
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
    
    # Load Gaussians
    print("Loading Gaussian PLY...")
    object_gaussians = load_ply(OBJECT_PATH)
    object_pcd = read_ply_to_point_cloud(OBJECT_PATH)
    l_max = object_gaussians['max_sh_degree']
    
    # Precompute SH sampling on GPU (done once)
    print("Precomputing SH basis on GPU...")
    directions, sqrt_weights, A, A_weighted, AT_A_weighted, AT_A = precompute_sh_sampling(num_samples, l_max, device)
    AT_A_inv = torch.linalg.inv(AT_A)
    
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
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device)
    transfer_start_time = time.time()
    
    with torch.no_grad():
        # ========================================================================
        # Step 1: Compute GLOBAL SH Coefficients for Environment Maps
        # ========================================================================
        print("Computing GLOBAL SH coefficients for environment maps...")
        
        sh_coeffs_source_global = compute_global_sh_coeffs(env_map_source_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)
        sh_coeffs_target_global = compute_global_sh_coeffs(env_map_target_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)
        sh_coeffs_offset = compute_global_sh_coeffs(offset_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)
        
        print(f"  Global SH source mean DC: {sh_coeffs_source_global[0].mean():.4f}")
        print(f"  Global SH target mean DC: {sh_coeffs_target_global[0].mean():.4f}")

        # ========================================================================
        # Step 2: Define Lambertian Kernel Coefficients (A_l)
        # ========================================================================
        A_l = torch.tensor([
            np.pi,                              # l=0 (1 coeff)
            2*np.pi/3, 2*np.pi/3, 2*np.pi/3,    # l=1 (3 coeffs)
            np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4,  # l=2 (5 coeffs)
            0., 0., 0., 0., 0., 0., 0.          # l=3 (7 coeffs, all zero)
        ], device=device, dtype=torch.float32)  # [16]
        
        print(f"  Lambertian kernel A_l applied.")

        # ========================================================================
        # Step 3: Analytic Irradiance Computation (Geomerics Non-Linear L1)
        # ========================================================================
        print("Computing irradiance E(n) for all normals (Geomerics Non-Linear)...")
        
        normals_for_sh = all_normals.clone()
        normals_for_sh[:, 0] = -normals_for_sh[:, 0]
        
        # --- NEW: Use Geomerics / Hazel 2015 robust diffuse transfer ---
        E_source = compute_nonlinear_irradiance(sh_coeffs_source_global, normals_for_sh, A_l, device)
        E_target = compute_nonlinear_irradiance(sh_coeffs_target_global, normals_for_sh, A_l, device)
        
        # ========================================================================
        # Step 4: Calculate Scalar Transfer Ratio (Irradiance Ratio)
        # ========================================================================
        epsilon = 1e-6
        T_diffuse = E_target / (E_source.abs() + epsilon)  # [N, 3]
        T_diffuse = torch.clamp(T_diffuse, min=-5.0, max=5.0)  # Prevent extreme values

        # ========================================================================
        # Step 5: Apply Transfer to Gaussian SH Coefficients
        # ========================================================================
        features_dc = object_gaussians['features_dc']  # [N, 1, 3]
        features_rest = object_gaussians['features_rest']  # [N, 15, 3]
        gaussians_sh_coeffs = torch.cat([features_dc, features_rest], dim=1)  # [N, num_coeffs, 3]

        print("\n--- Applying Transfer ---")
        
        # DIFFUSE PATH: Retain exact logic for the DC band
        print("1/2: Processing Diffuse DC Band...")
        gaussians_diffuse_adjusted = apply_transfer_fused_gpu(
            gaussians_sh_coeffs, T_diffuse, sh_coeffs_offset, A, AT_A_inv, batch_size=20000
        )
        
        # SPECULAR PATH: Process the Rest bands via the BRDF-free convolution without self-shadows
        print("2/2: Processing Specular Rest Bands...")
        gaussians_specular_rest = apply_specular_transfer_gpu(
            sh_coeffs=gaussians_sh_coeffs, 
            normals=all_normals, 
            sh_offset=sh_coeffs_offset, 
            A=A, 
            AT_A_inv=AT_A_inv, 
            directions=directions, 
            L_source_global=sh_coeffs_source_global, 
            L_target_global=sh_coeffs_target_global, 
            l_max=l_max, 
            device=device, 
            batch_size=5000
        )
        
        # Split back
        features_dc_adjusted = gaussians_diffuse_adjusted[:, 0:1, :]
        features_rest_adjusted = gaussians_specular_rest
        
        # Update gaussians
        object_gaussians['features_dc'] = features_dc_adjusted
        object_gaussians['features_rest'] = features_rest_adjusted
    
    # ========================================================================
    # END TIMING: Radiance Transfer Computation
    # ========================================================================
    transfer_end_time = time.time()
    transfer_elapsed = transfer_end_time - transfer_start_time
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