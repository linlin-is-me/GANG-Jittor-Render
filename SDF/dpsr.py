
import jittor as jt
from jittor import nn
from SDF.utils import spec_gaussian_filter, fftfreqs, img, grid_interp, point_rasterize
from SDF.fft3d import fft3d, hermitian_expand_last_axis
import numpy as np

class DPSR(nn.Module):
    def __init__(self, res, sig=10, scale=True, shift=True):
        """
        :param res: tuple of output field resolution. eg., (128,128)
        :param sig: degree of gaussian smoothing
        """
        super(DPSR, self).__init__()
        self.res = res
        self.sig = sig
        self.dim = len(res)
        self.denom = np.prod(res)
        G = spec_gaussian_filter(res=res, sig=sig).float()
        # self.G.requires_grad = False # True, if we also make sig a learnable parameter
        self.omega = fftfreqs(res, dtype=jt.float32)
        self.scale = scale
        self.shift = shift
        # Jittor: register_buffer not available; plain attr auto-registers as parameter
        self.G = G
        
    def execute(self, V, N):
        """
        :param V: (batch, nv, 2 or 3) tensor for point cloud coordinates
        :param N: (batch, nv, 2 or 3) tensor for point normals
        :return phi: (batch, res, res, ...) tensor of output indicator function field
        """
        assert(V.shape == N.shape) # [b, nv, ndims]
        ras_p = point_rasterize(V, N, self.res)  # [b, n_dim, dim0, dim1, dim2]

        batch, n_dim = ras_p.shape[:2]
        ras_complex = jt.stack([ras_p, jt.zeros_like(ras_p)], dim=-1)
        ras_s = fft3d(ras_complex.reshape((batch * n_dim, *self.res, 2)))
        half_width = self.res[-1] // 2 + 1
        ras_s = ras_s[:, :, :, :half_width]
        ras_s = ras_s.reshape((batch, n_dim, self.res[0], self.res[1], half_width, 2))
        ras_s = ras_s.permute([0, 2, 3, 4, 1, 5])

        # Gaussian filter is real and has shape [D,H,W,1,1].
        filtered = ras_s * self.G
        omega = self.omega.unsqueeze(-1) * (2.0 * np.pi)  # [D,H,W,n_dim,1]

        # -i * (a + ib) = b - ia.  Summing over the vector dimension gives
        # the Fourier-domain divergence as a real/imaginary pair.
        minus_i_filtered = jt.stack([filtered[..., 1], -filtered[..., 0]], dim=-1)
        div_n = jt.sum(minus_i_filtered * omega, dim=-2)
        lap = -jt.sum(omega ** 2, dim=-2)
        phi_freq = div_n / (lap + 1e-6)

        half_res = (self.res[0], self.res[1], half_width)
        dc_mask_np = np.ones(half_res, dtype=np.float32)
        dc_mask_np[(0,) * self.dim] = 0.0
        dc_mask = jt.array(dc_mask_np).unsqueeze(0).unsqueeze(-1)
        dc_mask.requires_grad = False
        phi_freq = phi_freq * dc_mask
        phi_freq_full = hermitian_expand_last_axis(phi_freq, self.res[-1])
        phi = fft3d(phi_freq_full, inverse=True)[..., 0]
        
        if self.shift or self.scale:
            # ensure values at points are zero
            fv = grid_interp(phi.unsqueeze(-1), V, batched=True).squeeze(-1) # [b, nv]
            if self.shift: # offset points to have mean of 0
                offset = jt.mean(fv, dim=-1)  # [b,] 
                phi = phi - offset.view(*tuple([-1] + [1] * self.dim))
                
            phi = phi.permute(*tuple([list(range(1,self.dim+1)) + [0]]))
            fv0 = phi[tuple([0] * self.dim)]  # [b,]
            phi = phi.permute(*tuple([[self.dim] + list(range(self.dim))]))
            
            if self.scale:
                phi = -phi / jt.abs(fv0.view(*tuple([-1]+[1] * self.dim))) *0.5
        return phi
