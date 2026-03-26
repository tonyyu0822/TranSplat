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
ENV_MAP_TARGET_PATH = '/scratch/shared/by12/Transplat/envmap/CombinedPano/garage/garage_enhance80.hdr'
OBJECT_PATH = '/scratch/shared/by12/Transplat/code/2d-gaussian-splatting/output/lego/point_cloud/iteration_30000/vanilla_point_cloud.ply'
RELIGHT_OUTPUT_PATH = "/scratch/shared/by12/Transplat/code/2d-gaussian-splatting/output/lego/point_cloud/iteration_30000/SHADOW_city_to_garage_point_cloud.ply"

# ============================================================================
# 2DGS Mock Classes & Camera Helpers
# ============================================================================
class MockGaussianModel:
    """Duck-types the official GaussianModel so we can pass our dict straight into the renderer."""
    def __init__(self, gaussians_dict):
        self.g = gaussians_dict
        self.active_sh_degree = gaussians_dict['max_sh_degree']
        
    @property
    def get_xyz(self): return self.g['xyz']
    @property
    def get_features(self): return torch.cat([self.g['features_dc'], self.g['features_rest']], dim=1)
    @property
    def get_opacity(self): return torch.sigmoid(self.g['opacity']) # Renderer expects activated [0,1]
    @property
    def get_scaling(self): return torch.exp(self.g['scaling'])     # Renderer expects activated scale
    @property
    def get_rotation(self): return F.normalize(self.g['rotation'], dim=1) # Normalized Quat
    
class MockPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False
    depth_ratio = 0.0

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = z_sign
    return P

class MiniCam:
    def __init__(self, W, H, fovX, fovY, znear, zfar, W2C, full_proj, origin):
        self.image_width = W
        self.image_height = H
        self.FoVx = fovX
        self.FoVy = fovY
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = W2C
        self.full_proj_transform = full_proj
        self.camera_center = origin

def create_telephoto_camera(scene_center, direction, radius, W, H, device):
    """
    Creates a perspective camera placed at a distance to encompass the scene.
    P is transposed for proper right-multiplication in the 2DGS CUDA rasterizer.
    """
    D = radius * 5.0
    origin = scene_center + direction * D
    
    forward = F.normalize(scene_center - origin, dim=0) # essentially -direction
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    if abs(torch.dot(forward, up)) > 0.99:
        up = torch.tensor([1.0, 0.0, 0.0], device=device)
        
    right = F.normalize(torch.cross(up, forward), dim=0)
    true_up = F.normalize(torch.cross(forward, right), dim=0)
    
    # Standard 3DGS View Space: X Right, Y Down, Z Forward
    R = torch.stack([right, -true_up, forward], dim=1)
    t = -origin @ R
    
    W2C = torch.eye(4, device=device)
    W2C[:3, :3] = R
    W2C[3, :3] = t
    
    fovY = 2.0 * math.asin(radius / D) * 1.2
    fovX = fovY
    
    znear = D - radius * 1.5
    zfar = D + radius * 1.5
    
    # Transposing P is CRITICAL because the rasterizer right-multiplies p @ full_proj
    P = getProjectionMatrix(znear, zfar, fovX, fovY).transpose(0, 1).to(device)
    full_proj = W2C @ P
    
    return MiniCam(W, H, fovX, fovY, znear, zfar, W2C, full_proj, origin), W2C, full_proj

# ============================================================================
# PLY I/O Functions
# ============================================================================
def load_ply(path, normal_b=True):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])),  axis=1)
    if normal_b:
        normal = np.stack((np.asarray(plydata.elements[0]["nx"]), np.asarray(plydata.elements[0]["ny"]), np.asarray(plydata.elements[0]["nz"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")], key = lambda x: int(x.split('_')[-1]))
    max_sh_degree = int(math.sqrt(len(extra_f_names) // 3 + 1)) - 1
    
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")], key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = sorted([p.name for p in plydata.elements[0].properties if p.name.startswith("rot")], key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    gaussians = {
        "xyz": nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda")),
        "features_dc": nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()),
        "features_rest": nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()),
        "opacity": nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda")),
        "scaling": nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda")),
        "rotation": nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda")),
        "normal": nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda")) if normal_b else None,
        "max_sh_degree": max_sh_degree
    }
    return gaussians

def save_ply_from_dict(dict, save_path):
    def construct_list_of_attributes(_features_dc, _features_rest, _scaling, _rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(_features_dc.shape[1]*_features_dc.shape[2]): l.append('f_dc_{}'.format(i))
        for i in range(_features_rest.shape[1]*_features_rest.shape[2]): l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(_scaling.shape[1]): l.append('scale_{}'.format(i))
        for i in range(_rotation.shape[1]): l.append('rot_{}'.format(i))
        return l
    
    xyz = dict['xyz'].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = dict['features_dc'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = dict['features_rest'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = dict['opacity'].detach().cpu().numpy()
    scale = dict['scaling'].detach().cpu().numpy()
    rotation = dict['rotation'].detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(dict['features_dc'], dict['features_rest'], dict['scaling'], dict['rotation'])]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_path)

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

def srgb_to_linear_torch(c_srgb):
    return torch.where(c_srgb <= 0.04045, c_srgb / 12.92, torch.pow((c_srgb + 0.055) / 1.055, 2.4))

def linear_to_srgb_torch(c_linear):
    return torch.where(c_linear <= 0.0031308, c_linear * 12.92, 1.055 * torch.pow(c_linear.clamp(min=1e-10), 1.0 / 2.4) - 0.055)

@torch.no_grad()
def apply_transfer_fused_gpu(sh_coeffs, T_diffuse, sh_offset, A, AT_A_inv, batch_size=20000):
    N, num_coeffs = sh_coeffs.shape[0], A.shape[1]
    adjusted = torch.zeros(N, num_coeffs, 3, device=sh_coeffs.device, dtype=sh_coeffs.dtype)
    
    # Gen by Cursor
    # Precompute the inversion and projection matrices combination
    # AT_A_inv is [K, K], A.T is [K, S]. M is [K, S]
    M = torch.matmul(AT_A_inv, A.T)
    
    for start in tqdm(range(0, N, batch_size), desc="Fused Transfer"):
        end = min(start + batch_size, N)
        batch = sh_coeffs[start:end] + sh_offset
        batch_T = T_diffuse[start:end]
        
        colors = srgb_to_linear_torch(torch.matmul(A, batch))
        colors = linear_to_srgb_torch(colors * batch_T.unsqueeze(1))
        
        adjusted[start:end] = torch.matmul(M, colors) - sh_offset
    return adjusted

def rotate_normals_torch(normals, rotation_matrix, device):
    return (torch.from_numpy(rotation_matrix).float().to(device) @ normals.T).T

def rotate_env_map(env_map, rotation_matrix):
    H, W, _ = env_map.shape
    lon, lat = np.meshgrid(np.linspace(0, 2 * np.pi, W, endpoint=False), np.linspace(0, np.pi, H))
    coords = np.stack([-np.sin(lat) * np.cos(lon), np.sin(lat) * np.sin(lon), np.cos(lat)], axis=-1)
    
    rotated_coords = (rotation_matrix.T @ coords.reshape(-1, 3).T).T.reshape(H, W, 3)
    lon_rot = np.arctan2(rotated_coords[..., 1], rotated_coords[..., 0])
    lon_rot[lon_rot < 0] += 2 * np.pi
    lat_rot = np.arccos(rotated_coords[..., 2]) 
    
    u = np.clip((lon_rot / (2 * np.pi) * W).astype(np.float32), 0, W - 1)
    v = np.clip((lat_rot / np.pi * H).astype(np.float32), 0, H - 1)
    return cv2.remap(env_map, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# ============================================================================
# NEW: 2DGS Rasterizer Shadow Mapping for V_lm
# ============================================================================
import os
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm

def get_scene_bounds(gaussians):
    """
    Calculates the scene center and radius directly from the Gaussian point cloud.
    """
    xyz = gaussians['xyz'].detach()
    scene_center = xyz.mean(dim=0)
    
    # Use 99th percentile to be robust against outlier floats, or just max.
    # Here max is usually fine for a trained 3DGS bound.
    scene_radius = torch.max(torch.norm(xyz - scene_center, dim=1)).item()
    
    # Add a small padding
    scene_radius *= 1.8
    return scene_center, scene_radius

def compute_visibility_sh_2dgs(gaussians, directions, A_weighted, AT_A_weighted, sqrt_weights, device, object_path, res=512):
    """
    Computes SH occlusion using the 2DGS rasterizer to render internal depth maps.
    Uses ground-truth training cameras to establish deterministic scene bounds.
    """
    try:
        from gaussian_renderer import render
    except ImportError:
        raise ImportError("Could not import 'render' from 'gaussian_renderer'. Run from 2DGS root.")

    N = gaussians['xyz'].shape[0]
    S = directions.shape[0]
    K = A_weighted.shape[1]
    
    xyz = gaussians['xyz'].detach()
    
    print("Calculating deterministic scene bounds from object point cloud...")
    scene_center, scene_radius = get_scene_bounds(gaussians)
    print(f"  -> True Scene Radius: {scene_radius:.4f}")
    
    mock_gaussians = MockGaussianModel(gaussians)
    pipe = MockPipeline()
    bg_color = torch.zeros(3, dtype=torch.float32, device=device)
    gaussian_scales = mock_gaussians.get_scaling.detach() 
    max_scales, _ = torch.max(gaussian_scales, dim=1) # [N]
    
    # A Gaussian's influence practically ends around 3 standard deviations.
    # We use this exact physical footprint as its depth bias.
    per_gaussian_bias = max_scales * 3.0 
    
    # Gen by Cursor
    # Pre-allocate SH accumulation matrix directly to avoid massive [N, S] 
    # visibility matrix memory footprint (~4GB for 1M points).
    AT_b = torch.zeros((K, N), dtype=torch.float32, device=device)

    # Gen by Cursor
    # Pre-calculate constant camera projection intrinsics
    # (Since D = radius * 5.0, FoV is constant across all views)
    fov_const = 2.0 * math.asin(1.0 / 5.0) * 1.2
    f_const = 1.0 / math.tan(fov_const / 2.0)
    half_res = (res - 1) * 0.5

    print(f"Rendering {S} depth maps for global occlusion using 2DGS rasterizer...")
    for i in tqdm(range(S), desc="Shadow Mapping"):
        dir_vec = directions[i]
        
        # Spawn pseudo-orthographic camera tightly bound to the true camera orbit
        cam, W2C, full_proj = create_telephoto_camera(scene_center, dir_vec, scene_radius, res, res, device)
        
        # Render Depth Map
        with torch.no_grad():
            render_pkg = render(cam, mock_gaussians, pipe, bg_color)
            
            # 2DGS exports surface depth, 3DGS defaults to expected depth
            if "surf_depth" in render_pkg:
                depth_map = render_pkg["surf_depth"][0] # [H, W]
            elif "depth" in render_pkg:
                depth_map = render_pkg["depth"][0]
            else:
                raise ValueError("Rasterizer didn't return depth (expected 'surf_depth' or 'depth')")
                
        # Gen by Cursor
        # Transform Points to shadow camera space using a single GEMM for speed
        xyz_view = torch.matmul(xyz, W2C[:3, :3]) + W2C[3, :3]
        depth_gaussian = xyz_view[:, 2] # Actual depth from camera
        
        # Project points to NDC using intrinsics directly
        ndc_x = (xyz_view[:, 0] * f_const) / depth_gaussian
        ndc_y = (xyz_view[:, 1] * f_const) / depth_gaussian
        
        # NDC -> UV
        half_res = (res - 1) * 0.5
        u_idx = torch.clamp(((ndc_x + 1.0) * half_res + 0.5).long(), 0, res - 1)
        v_idx = torch.clamp(((ndc_y + 1.0) * half_res + 0.5).long(), 0, res - 1)
        
        sampled_depths = depth_map[v_idx, u_idx]
        is_occluded = (depth_gaussian > sampled_depths + per_gaussian_bias) & (sampled_depths > 0)
        
        # Accumulate visibility directly into Spherical Harmonics (V_lm)
        # to save GPU memory and bandwidth (no need to store N x S matrix)
        vis_mask = (~is_occluded).float()
        v_s = vis_mask * sqrt_weights[i]
        AT_b += torch.outer(A_weighted[i], v_s) # [K, N] incremental accumulation
        
        # --- Visualization (Save first sample) ---
        if i == 0:
            depth_np = depth_map.detach().cpu().numpy()
            if depth_np.max() > 0:
                depth_vis = (depth_np - depth_np[depth_np > 0].min()) / (depth_np.max() - depth_np[depth_np > 0].min() + 1e-5)
                depth_vis = np.clip(depth_vis * 255, 0, 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
                cv2.imwrite("shadow_depth_map_sample.png", depth_vis)
                print(f"\n  -> Saved visualization to 'shadow_depth_map_sample.png'")
                
    print("Projecting visibility matrix to Spherical Harmonics (V_lm)...")
    V_lm = torch.linalg.solve(AT_A_weighted, AT_b).T              # [N, K]
    
    return V_lm

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    num_samples = 1000
    rotation_matrix = np.eye(3)
    
    print("Loading Gaussian PLY...")
    object_gaussians = load_ply(OBJECT_PATH)
    l_max = object_gaussians['max_sh_degree']
    
    print("Precomputing SH basis & Gaunt Tensor on GPU...")
    directions, sqrt_weights, A, A_weighted, AT_A_weighted, AT_A, gaunt_tensor = precompute_sh_sampling(num_samples, l_max, device)
    AT_A_inv = torch.linalg.inv(AT_A)
    
    print("Loading environment maps...")
    env_map_source = cv2.cvtColor(cv2.imread(ENV_MAP_SOURCE_PATH, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    env_map_target = cv2.cvtColor(cv2.imread(ENV_MAP_TARGET_PATH, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if env_map_target.shape[:2] != env_map_source.shape[:2]:
        env_map_target = cv2.resize(env_map_target, (env_map_source.shape[1], env_map_source.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    if torch.all(object_gaussians['normal'] == 0):
        with torch.no_grad():
            object_gaussians['normal'] = quaternion2rotmat(F.normalize(object_gaussians['rotation'], dim=1))[..., 2]

    all_normals = rotate_normals_torch(object_gaussians['normal'], rotation_matrix, device)
    all_normals = all_normals / all_normals.norm(dim=1, keepdim=True)
    env_map_source = rotate_env_map(env_map_source, rotation_matrix)
    
    env_map_source_gpu = torch.from_numpy(env_map_source.astype(np.float32)).to(device)
    env_map_target_gpu = torch.from_numpy(env_map_target.astype(np.float32)).to(device)
    offset_gpu = torch.from_numpy(env_map_target.astype(np.float32) * 0 + 0.5).to(device)
    
    # ========================================================================
    # START TIMING: Radiance Transfer Computation
    # ========================================================================
    print("\n" + "="*70)
    print("Starting SH-Space Precomputed Radiance Transfer...")
    print("="*70)
    torch.cuda.reset_peak_memory_stats(device)
    transfer_start_time = time.time()
    
    with torch.no_grad():
        print("Computing L_lm for environments...")
        L_lm_source = compute_global_sh_coeffs(env_map_source_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)
        L_lm_target = compute_global_sh_coeffs(env_map_target_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)
        sh_coeffs_offset = compute_global_sh_coeffs(offset_gpu, directions, sqrt_weights, A_weighted, AT_A_weighted)
        
        # --- NEW: Use 2DGS GPU shadow mapping instead of Poisson mesh ---
        V_lm = compute_visibility_sh_2dgs(
            object_gaussians, directions, A_weighted, AT_A_weighted, sqrt_weights, device, OBJECT_PATH, res=1024
        ) 

        print("Multiplying L_lm and V_lm via Gaunt Tensor (Clebsch-Gordan)...")
        L_shad_source = torch.einsum('ijk,ic,nj->nkc', gaunt_tensor, L_lm_source, V_lm) 
        L_shad_target = torch.einsum('ijk,ic,nj->nkc', gaunt_tensor, L_lm_target, V_lm)

        A_l = torch.tensor([np.pi, 2*np.pi/3, 2*np.pi/3, 2*np.pi/3, np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4, 0., 0., 0., 0., 0., 0., 0.], device=device, dtype=torch.float32)
        
        print("Computing shadowed irradiance E(n)...")
        normals_for_sh = all_normals.clone()
        normals_for_sh[:, 0] = -normals_for_sh[:, 0]
        
        Y_hat = evaluate_sh_bases_torch(normals_for_sh, l_max, device) * A_l.view(1, -1)
        E_source = torch.einsum('nk,nkc->nc', Y_hat, L_shad_source)
        E_target = torch.einsum('nk,nkc->nc', Y_hat, L_shad_target)
        
        # ========================================================================
        # Step 4: Calculate Stabilized RGB Transfer Ratio with SH Ambient Occlusion
        # ========================================================================
        E_source_clean = torch.clamp(E_source, min=0.0)
        E_target_clean = torch.clamp(E_target, min=0.0)
        
        # 1. Extract Ambient Occlusion from the DC band of V_lm
        # The integral of the DC basis function over the sphere is ~3.5449.
        # Dividing by this normalizes the AO to a [0, 1] range (1 = fully open sky, 0 = fully enclosed)
        AO = torch.clamp(V_lm[:, 0:1] / 3.5449, min=0.0, max=1.0) # [N, 1]
        
        # 2. Define a base ambient intensity for the source scene (e.g., 10% of mean light)
        base_ambient = 0.10 * E_source_clean.mean() 
        
        # 3. Compute spatially varying ambient light (Epsilon)
        # Deep crevices get near-zero ambient light, exposed areas get full ambient light
        local_ambient = base_ambient * AO + 1e-4 # [N, 1]
        
        # 4. Perform RGB division (Un-bake source, apply target)
        T_diffuse = E_target_clean / (E_source_clean + local_ambient)
        
        # 5. Soft clamp to prevent extreme highlight blowouts
        T_diffuse = torch.clamp(T_diffuse, min=0.0, max=3.0)
        # -----------------------------------------------

        gaussians_sh_coeffs = torch.cat([object_gaussians['features_dc'], object_gaussians['features_rest']], dim=1)
        gaussians_sh_coeffs_adjusted = apply_transfer_fused_gpu(gaussians_sh_coeffs, T_diffuse, sh_coeffs_offset, A, AT_A_inv, batch_size=20000)
        
        object_gaussians['features_dc'] = gaussians_sh_coeffs_adjusted[:, 0:1, :]
        object_gaussians['features_rest'] = gaussians_sh_coeffs_adjusted[:, 1:, :]
    
    transfer_end_time = time.time()
    transfer_elapsed = transfer_end_time - transfer_start_time
    mem_peak = torch.cuda.max_memory_allocated(device)
    print("\n" + "="*70)
    print(f"Radiance transfer completed in {transfer_elapsed:.2f} seconds ({transfer_elapsed/60:.2f} minutes)")
    print(f"GPU Peak memory during computation: {mem_peak / 1024**2:.2f} MB")
    print("="*70 + "\n")
    
    save_ply_from_dict(object_gaussians, RELIGHT_OUTPUT_PATH)
    print("Done!")

"""
python radiance_transfer_ambient_occlusion.py

"""