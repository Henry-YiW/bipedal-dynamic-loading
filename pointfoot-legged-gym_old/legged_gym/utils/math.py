# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

# @ torch.jit.script
def quat_rotate_inverse(q, v):
    """
    Rotate vector(s) v by quaternion(s) q^-1.
    
    Args:
        q: quaternion(s) of shape (..., 4)
        v: vector(s) of shape (..., 3)
    
    Returns:
        The rotated vector(s) of shape (..., 3)
    """
    shape = q.shape[:-1]
    q_w, q_x, q_y, q_z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Compute the quaternion conjugate
    q_conj = torch.stack([q_w, -q_x, -q_y, -q_z], dim=-1)
    
    # Create a quaternion with the vector as its imaginary part
    v_quat = torch.cat([torch.zeros(*shape, 1, device=v.device), v], dim=-1)
    
    # Apply q^-1 * v * q
    # First, q^-1 * v
    temp_quat = quat_mul(q_conj, v_quat)
    # Then, (q^-1 * v) * q
    result_quat = quat_mul(temp_quat, q)
    
    # Return the imaginary part, which is the rotated vector
    return result_quat[..., 1:]

# @ torch.jit.script
def quat_mul(a, b):
    """
    Multiply quaternion(s) a and b.
    
    Args:
        a: quaternion(s) of shape (..., 4)
        b: quaternion(s) of shape (..., 4)
    
    Returns:
        The product quaternion(s) of shape (..., 4)
    """
    a_w, a_x, a_y, a_z = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b_w, b_x, b_y, b_z = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    
    # Hamilton product
    w = a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z
    x = a_w * b_x + a_x * b_w + a_y * b_z - a_z * b_y
    y = a_w * b_y - a_x * b_z + a_y * b_w + a_z * b_x
    z = a_w * b_z + a_x * b_y - a_y * b_x + a_z * b_w
    
    return torch.stack([w, x, y, z], dim=-1)