import torch
import numpy as np

def quaternion2rotmat(q):
    r, x, y, z = q.split(1, -1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)
        ], -1).reshape([len(q), 3, 3])
    return R

def quat_to_rotmat_cpp(quat):
    # This is replicating the C++ implementation. Note w,x,y,z is r,x,y,z
    w, x, y, z = quat.split(1, -1)
    # The C++ code is constructing a glm::mat3 which is COLUMN-MAJOR!
    R = torch.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y)
    ], -1).reshape([len(quat), 3, 3]).transpose(1, 2)
    return R

q = torch.tensor([[1.0, 0.4, 0.2, 0.1], [0.5, 0.5, 0.5, 0.5]])
q = torch.nn.functional.normalize(q, dim=1)

mat_py = quaternion2rotmat(q)
mat_cpp = quat_to_rotmat_cpp(q)

print("Python:\n", mat_py[0])
print("C++ translated:\n", mat_cpp[0])
print("Match?", torch.allclose(mat_py, mat_cpp))
print("Difference max:", (mat_py - mat_cpp).abs().max().item())

print("Python Z-axis:\n", mat_py[0, :, 2])
print("C++ Z-axis:\n", mat_cpp[0, :, 2])
