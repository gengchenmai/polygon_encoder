import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from math import pi

from polygonembed.ddsl_utils import *



class SimplexFT(Function):
    """
    Fourier transform for signal defined on a j-simplex set in R^n space
    :param V: vertex tensor. float tensor of shape (n_vertex, n_dims)
    :param E: element tensor. int tensor of shape (n_elem, j or j+1)
              if j cols, triangulate/tetrahedronize interior first.
    :param D: int ndarray of shape (n_elem, n_channel)
    :param res: n_dims int tuple of number of frequency modes
    :param t: n_dims tuple of period in each dimension
    :param j: dimension of simplex set
    :param mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
    :return: F: ndarray of shape (res[0], res[1], ..., res[-1]/2, n_channel)
                last dimension is halfed since the signal is assumed to be real
    """
    @staticmethod
    def forward(ctx, V, E, D, res, t, j,  
                min_freqXY, max_freqXY, mid_freqXY = None, freq_init = "fft",
                elem_batch=100, mode='density'):
        '''
        Args:
            V: vertex tensor. float tensor of shape (batch_size, num_vert, n_dims = 2)
            E: element tensor. int tensor of shape (batch_size, n_elem = num_vert, j or j+1)
                if j cols, triangulate/tetrahedronize interior first.
                (num_vert, 2), indicate the connectivity


            D: int ndarray of shape (batch_size, n_elem = num_vert, n_channel = 1), all 1
            res: n_dims int tuple of number of frequency modes, (fx, fy) for polygon
            t: n_dims tuple of period in each dimension, (tx, ty) for polygon
            j: dimension of simplex set, 2 for polygon
            max_freqXY: the maximum frequency
            min_freqXY: the minimum frequency
            freq_init: frequency generated method, 
                "geometric": geometric series
                "fft": fast fourier transformation
            mode: normalization mode.
                 'density' for preserving density, 'mass' for preserving mass
        Return:
            F: ndarray of shape (res[0], res[1], ..., res[-1]/2, n_channel)
                last dimension is halfed since the signal is assumed to be real
                Shape (fx, fy//2+1, n_channel = 1, 2) for polygon case
        '''
        ## check if E is subdim
        '''
        subdim: See ddsl_illustration.jpg
            True: V is a sequence of vertex and E is sequence of lines
            False: V is a sequence of vertex + one and E is sequence of triangles
        '''
        assert V.shape[0] == E.shape[0] == D.shape[0] # batch_size
        batch_size = V.shape[0]
        subdim = E.shape[-1] == j and V.shape[-1] == j
        assert (E.shape[-1] == j+1 or subdim)
        # assert V.shape[0] == E.shape[0] == D.shape[0] # batch_size dim should match
        
        if subdim:
            # make sure all D has same density
            D_repeat = torch.repeat_interleave(D[:, 0].unsqueeze(1), repeats = D.shape[1], dim=1)
            assert((D == D_repeat).sum().item() == D.numel()) # assert same densities for all simplices (homogeneous filling)
            # add (0,0) as an auxilary vertex in V
            # aux_vert_mat: shape (batch_size, 1, n_dims = 2)
            aux_vert_mat = torch.zeros((batch_size, 1, V.shape[-1]), device=V.device, dtype=V.dtype)
            # V: (batch_size, num_vert + 1, n_dims = 2)
            V = torch.cat((V, aux_vert_mat), dim=1)

            # the index of the added auxilary vertex (0, 0)
            aux_vert_idx = V.shape[1] - 1
            # add_aux_vert_mat: (batch_size, n_elem = num_vert, 1)
            # values are the index of the added auxilary vertex
            add_aux_vert_mat = torch.zeros((batch_size, E.shape[1], 1), device=E.device, dtype=E.dtype) + aux_vert_idx
            # E: (batch_size, n_elem = num_vert, j+1 = 3), add (0, 0) as the 3rd vertice for each line in E, we have construct all 2-simplex
            E = torch.cat((E, add_aux_vert_mat), dim=-1)
        
        n_elem = E.shape[-2] # n_elem = num_vert,  number of 2-simplex, the triangles
        n_vert = V.shape[-2] # n_vert = num_vert + 1, number of vertex
        n_channel = D.shape[-1] # n_channel = 1

        ## save context info for backwards
        ctx.mark_non_differentiable(E, D) # mark non-differentiable
        ctx.res = res
        ctx.t = t
        ctx.j = j
        ctx.mode = mode
        ctx.n_dims = V.shape[-1] # n_dims = 2
        ctx.elem_batch = elem_batch # elem_batch = 100
        ctx.subdim = subdim

        ctx.min_freqXY = min_freqXY
        ctx.max_freqXY = max_freqXY
        ctx.mid_freqXY = mid_freqXY,
        ctx.freq_init = freq_init


        
        
        # compute content array
        # C: normalized simple content, \gama_n^j, shape (batch_size, n_elem = num_vert, 1)
        # unsigned: Equation 6 in https://openreview.net/pdf?id=B1G5ViAqFm
        # signed: Equation 8 in https://openreview.net/pdf?id=B1G5ViAqFm
        C = math.factorial(j) * simplex_content(V, E, signed=subdim) # [n_elem, 1]
        ctx.save_for_backward(V, E, D, C)

        ## compute frequencies F
        n_dims = ctx.n_dims
        assert(n_dims == len(res))  # consistent spacial dimensionality
        assert(E.shape[1] == D.shape[1])  # consistent vertex numbers
        assert(mode in ['density', 'mass'])


        
        # frequency tensor
        '''
        omega: the fft frequance matrix
            if extract  = True
                for res = (fx, fy) => shape (fx, fy//2+1, 2)
                for res = (f1, f2, ..., fn) => shape (f1, f2, ..., f_{n-1}, fn//2+1, n = n_dims)
        '''
        # omega = fftfreqs(res, dtype=V.dtype).to(V.device) # [dim0, dim1, dim2, d]
        # omega: [dim0, dim1, dim2, d]
        omega = get_fourier_freqs(res = res, 
                                    min_freqXY = min_freqXY, 
                                    max_freqXY = max_freqXY, 
                                    mid_freqXY = mid_freqXY,
                                    freq_init = freq_init, 
                                    dtype=V.dtype).to(V.device)

        # normalize frequencies
        for dim in range(n_dims):
            omega[..., dim] *= 2 * math.pi / t[dim]

        

        # initialize output F
        # F_shape = [fx, fy//2+1]
        F_shape = list(omega.shape)[:-1]
        # F_shape = [fx, fy//2+1, n_channel = 1, 2]
        F_shape += [n_channel, 2]
        # F_shape = [batch_size, fx, fy//2+1, n_channel = 1, 2]
        F_shape  = [batch_size] + F_shape
        # F: shape (batch_size, fx, fy//2+1, n_channel = 1, 2)
        F = torch.zeros(*F_shape, dtype=V.dtype, device=V.device) # [dimX, dimY, dimZ, n_chan, 2] 2: real/imag

        # compute element-point tensor
        # P: point tensor. float, shape (batch_size, n_elem = num_vert, j+1 = 3, n_dims = 2)
        # P = V[E]
        P = coord_P_lookup_from_V_E(V, E)

        

        # loop over element/simplex batches
        for idx in range(math.ceil(n_elem/elem_batch)):
            id_start = idx * elem_batch
            id_end = min((idx+1) * elem_batch, n_elem)
            # Xi: coordinate mat, shape [batch_size, elem_batch, j+1, n_dims = 2]
            Xi = P[:, id_start:id_end] 
            # Di: simple density mat, shape [batch_size, elem_batch, n_channel = 1]
            Di = D[:, id_start:id_end] 
            # Ci: normalized simple content, \gama_n^j, Equ. 6, shape [batch_size, elem_batch, 1]
            Ci = C[:, id_start:id_end] 
            # CDi: \gama_n^j * \rho_n
            # CDi: shape [batch_size, elem_batch, n_channel]
            CDi = Ci.expand_as(Di) * Di 
            # sig: shape (batch_size, elem_batch, j+1, fx, fy//2+1)
            # k dot x, Equation 3 in https://openreview.net/pdf?id=B1G5ViAqFm
            sig = torch.einsum('lbjd,...d->lbj...', (Xi, omega)) 


            # sig: shape (batch_size, elem_batch, j+1, fx, fy//2+1, 1)
            sig = torch.unsqueeze(sig, dim=-1) # [elem_batch, j+1, dimX, dimY, dimZ, 1]
            # esig: e^{-i*\sigma_t}, Euler's formular, 
            #      shape (batch_size, elem_batch, j+1, fx, fy//2+1, 1, 2), 
            esig = torch.stack((torch.cos(sig), -torch.sin(sig)), dim=-1) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 2]
            # sig: shape (batch_size, elem_batch, j+1, fx, fy//2+1, 1, 1)
            sig = torch.unsqueeze(sig, dim=-1) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 1]

            

            # denom: prod(sigma_t - sigm_l),  the demoniator of Equation 7 in https://openreview.net/pdf?id=B1G5ViAqFm
            # denom: shape (batch_size, elem_batch, j+1, fx, fy//2+1, 1, 1)
            denom = torch.ones_like(sig) # [elem_batch, j+1, dimX, dimY, dimZ, 1, 1]
            for dim in range(1, j+1):
                seq = permute_seq(dim, j+1)
                denom *= sig - sig[:, :, seq]
            # tmp: shape (batch_size, elem_batch, fx, fy//2+1, 1, 2)
            #       sum e^{-i*\sigma_t}/prod(sigma_t - sigm_l), 
            #      signed context, Equation 4 in https://openreview.net/pdf?id=B1G5ViAqFm
            tmp = torch.sum(esig / denom, dim=2) # [elem_batch, dimX, dimY, dimZ, 1, 2]

            # select all cases when denom == 0
            mask = ((denom == 0).repeat_interleave(repeats = 2, dim=-1).sum(dim = 2) > 0)
            # mask all cases as 0
            tmp[mask] = 0
            
            # CDi: shape (batch_size, elem_batch, n_channel, 1)
            CDi.unsqueeze_(-1) # [elem_batch, n_channel, 1]
            # CDi: shape (batch_size, elem_batch, 1, 1, n_channel, 1)
            for _ in range(n_dims): # unsqueeze to broadcast
                CDi.unsqueeze_(dim=2) # [elem_batch, 1, 1, 1, n_channel, 2]

            # shape_ = (batch_size, elem_batch, fx, fy//2+1, n_channel, 2)
            shape_ = (list(tmp.shape[:-2])+[n_channel, 2])
            # tmp: (batch_size, elem_batch, fx, fy//2+1, n_channel, 2)
            #      \gama_n^j * \rho_n * sum e^{-i*\sigma_t}/prod(sigma_t - sigm_l)
            tmp = tmp * CDi # [elem_batch, dimX, dimY, dimZ, n_channel, 2]
            # Fi: shape (batch_size, fx, fy//2+1, n_channel, 2)
            Fi = torch.sum(tmp, dim=1, keepdim=False) # [dimX, dimY, dimZ, n_channel, 2]
            
            
            # CDi_: shape (batch_size, 1, 1, n_channel, 1)
            #       the sum of the polygon content
            CDi_ = torch.sum(CDi, dim=1)
            # CDi_: shape (batch_size, n_channel, 1)
            for _ in range(n_dims): # squeeze dims
                CDi_.squeeze_(dim=1) # [n_channel, 2]

            
            # Fi[:, tuple([0] * n_dims)] = - 1 / factorial(j) * CDi_ # ?????

            # Fi: shape (batch_size, fx, fy//2+1, n_channel = 1, 2)
            if n_dims == 2:
                Fi[:, 0, 0] = - 1 / math.factorial(j) * CDi_
            elif n_dims == 3:
                Fi[:, 0, 0, 0] = - 1 / math.factorial(j) * CDi_
            else:
                raise Exception("n_dims is not 2 or 3")
            F += Fi

        # F: shape (batch_size, fx, fy//2+1, n_channel = 1, 2)
        # multiply tensor F by i ** j
        # see Equation 4 and 7 in https://openreview.net/pdf?id=B1G5ViAqFm
        F = img(F, deg=j) # Fi *= i**j [dimX, dimY, dimZ, n_chan, 2] 2: real/imag

        if mode == 'density':
            res_t = torch.tensor(res)
            if not torch.equal(res_t, res[0]*torch.ones(len(res), dtype=res_t.dtype)):
                print("WARNING: density preserving mode not correctly implemented if not all res are equal")
            F *= res[0] ** j
        return F


class DDSL_spec(nn.Module):
    """
    Module for DDSL layer. Takes in a simplex mesh and returns the spectral raster.
    """
    def __init__(self, res, t, j, 
        min_freqXY, max_freqXY, mid_freqXY = None, freq_init = "fft",
        elem_batch=100, mode='density'):
        """
        Args:
            res: n_dims int tuple of number of frequency modes
            t: n_dims tuple of period in each dimension
            j: dimension of simplex set
            max_freqXY: the maximum frequency
            min_freqXY: the minimum frequency
            freq_init: frequency generated method, 
                "geometric": geometric series
                "fft": fast fourier transformation
            elem_batch: element-wise batch size.
            mode: 'density' for density conserving, or 'mass' for mess conserving. Defaults 'density'
        """
        super(DDSL_spec, self).__init__()
        self.res = res
        self.t = t
        self.j = j
        self.min_freqXY = min_freqXY
        self.max_freqXY = max_freqXY
        self.mid_freqXY = mid_freqXY
        self.freq_init = freq_init
        self.elem_batch = elem_batch
        self.mode = mode
    def forward(self, V, E, D):
        """
        V: vertex tensor. float tensor of shape (batch_size, num_vert, n_dims = 2)
        E: element tensor. int tensor of shape (batch_size, n_elem = num_vert, j or j+1)
                if j cols, triangulate/tetrahedronize interior first.
                (num_vert, 2), indicate the connectivity
        D: int ndarray of shape (batch_size, n_elem, n_channel)
        :return F: ndarray of shape (res[0], res[1], ..., res[-1]/2, n_channel) 
                   last dimension is halfed since the signal is assumed to be real
                F: shape (batch_size, fx, fy//2+1, n_channel = 1, 2)
        """
        V, D = V.double(), D.double()
        return SimplexFT.apply(V,E,D,self.res,self.t,self.j,
                self.min_freqXY, self.max_freqXY, self.mid_freqXY, self.freq_init,
                self.elem_batch,self.mode)

    
class DDSL_phys(nn.Module):
    """
    Module for DDSL layer. Takes in a simplex mesh and returns a dealiased raster image (in physical domain).
    """
    def __init__(self, res, t, j, 
        min_freqXY, max_freqXY, mid_freqXY = None, freq_init = "fft",
        smoothing='gaussian', sig=2.0, elem_batch=100, mode='density'):
        """
        Args:
            res: n_dims int tuple of number of frequency modes
            t: n_dims tuple of period in each dimension
            j: dimension of simplex set
            max_freqXY: the maximum frequency
            min_freqXY: the minimum frequency
            freq_init: frequency generated method, 
                "geometric": geometric series
                "fft": fast fourier transformation
            smoothing: str, choice of spectral smoothing function. Defaults 'gaussian'
            sig: sigma of gaussian at highest frequency
            elem_batch: element-wise batch size.
            mode: 'density' for density conserving, or 'mass' for mess conserving. Defaults 'density'
        """
        super(DDSL_phys, self).__init__()
        self.res = res
        self.t = t
        self.j = j
        self.min_freqXY = min_freqXY
        self.max_freqXY = max_freqXY
        self.mid_freqXY = mid_freqXY
        self.freq_init = freq_init
        self.elem_batch = elem_batch
        self.mode = mode
        self.filter = None
        self.sig = sig

        if isinstance(smoothing, str):
            assert(smoothing in ["gaussian"])
            if smoothing == 'gaussian':
                # filter: shape (fx, fy//2+1, 1, 1)
                self.filter = self._gaussian_filter()

    def forward(self, V, E, D):
        """
        V: vertex tensor. float tensor of shape (batch_size, num_vert, n_dims = 2)
        E: element tensor. int tensor of shape (batch_size, n_elem = num_vert, j or j+1)
            if j cols, triangulate/tetrahedronize interior first.
            (num_vert, 2), indicate the connectivity
        :param D: int ndarray of shape (batch_size, n_elem, n_channel)
        Return: 
            f: dealiased raster image in physical domain of shape (batch_size, res[0], res[1], ..., res[-1], n_channel)
                shape (batch_size, fx, fy, n_channel = 1)
        """

        V, D = V.double(), D.double()
        # F: shape (batch_size, fx, fy//2+1, n_channel = 1, 2) for polygon case
        F = SimplexFT.apply(V,E,D,self.res,self.t,self.j,
                    self.min_freqXY, self.max_freqXY, self.mid_freqXY, self.freq_init,
                    self.elem_batch,self.mode)
        F[torch.isnan(F)] = 0 # pad nans to 0
        if self.filter is not None:
            # filter: shape (fx, fy//2+1, 1, 1)
            self.filter = self.filter.to(F.device)
            # filter_: shape (fx, fy//2+1, 1, 2)
            filter_ = torch.repeat_interleave(self.filter, repeats = F.shape[-1], dim=-1)
            # F: shape (batch_size, fx, fy//2+1, n_channel = 1, 2) for polygon case
            F *= filter_ # [dim0, dim1, dim2, n_channel, 2]
        dim = len(self.res)
        # F: shape (batch_size, n_channel = 1, fx, fy//2+1,  2) for polygon case
        F = F.permute(*([0, dim+1] + list(range(1, dim+1)) + [dim+2])) # [n_channel, dim0, dim1, dim2, 2]
        # f: shape (batch_size, n_channel = 1, fx, fy)
        f = torch.irfft(F, dim, signal_sizes=self.res)
        # f: shape (batch_size, fx, fy, n_channel = 1)
        f = f.permute(*([0] + list(range(2, 2+dim)) + [1]))
                        
        return f
    
    def _gaussian_filter(self):
        '''
        Return:
            filter_: shape (fx, fy//2+1, 1, 1)
        '''
        
        # omega = fftfreqs(self.res, dtype=torch.float64) # [dim0, dim1, dim2, d]

        # omega: shape (fx, fy//2+1, 2)
        omega = get_fourier_freqs(res = self.res, 
                                    min_freqXY = self.min_freqXY, 
                                    max_freqXY = self.max_freqXY, 
                                    mid_freqXY = self.mid_freqXY,
                                    freq_init = self.freq_init, 
                                    dtype=torch.float64)
        # dis: shape (fx, fy//2+1)
        dis = torch.sqrt(torch.sum(omega ** 2, dim=-1))
        # filter_: shape (fx, fy//2+1, 1, 1)
        filter_ = torch.exp(-0.5*((self.sig*2*dis/self.res[0])**2)).unsqueeze(-1).unsqueeze(-1)
        filter_.requires_grad = False
        return filter_