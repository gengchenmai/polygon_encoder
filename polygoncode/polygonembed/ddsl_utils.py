import torch
import numpy as np
import math
from math import factorial
import os


def get_fourier_freqs(res, min_freqXY, max_freqXY, mid_freqXY = None, freq_init = "geometric", dtype=torch.float32, exact=True, eps = 1e-4):
    """
    Helper function to return frequency tensors
    This is a generalization of fftfreqs()
    Args:
        res: n_dims int tuple of number of frequency modes, (fx, fy) for polygon
        max_freqXY: the maximum frequency
        min_freqXY: the minimum frequency
        freq_init: frequency generated method, 
            "geometric": geometric series
            "fft": fast fourier transformation
    :return: omega: 
                if extract  = True
                    for res = (fx, fy) => shape (fx, fy//2+1, 2)
                    for res = (f1, f2, ..., fn) => shape (f1, f2, ..., f_{n-1}, fn//2+1, n)
                if extract  = False
                    for res = (fx, fy) => shape (fx, fy//2, 2)
                    for res = (f1, f2, ..., fn) => shape (f1, f2, ..., f_{n-1}, fn//2, n)
    """
    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = make_fourier_freq_vector(res_dim = r_, 
                                        min_freqXY = min_freqXY, 
                                        max_freqXY = max_freqXY, 
                                        mid_freqXY = mid_freqXY,
                                        freq_init = freq_init, 
                                        get_full_vector = True)
        freqs.append(torch.tensor(freq, dtype=dtype))
    r_ = res[-1]
    freq = make_fourier_freq_vector(res_dim = r_, 
                                    min_freqXY = min_freqXY, 
                                    max_freqXY = max_freqXY, 
                                    mid_freqXY = mid_freqXY,
                                    freq_init = freq_init, 
                                    get_full_vector = False)
    if exact:
        freqs.append(torch.tensor(freq, dtype=dtype))
    else:
        freqs.append(torch.tensor(freq[:-1], dtype=dtype))
    omega = torch.meshgrid(freqs)
    omega = list(omega)
    omega = torch.stack(omega, dim=-1)

    omega[0, 0, :] = torch.FloatTensor(np.random.rand(2)*eps).to(omega.dtype)
    return omega


def compute_geoemtric_series(min_, max_, num):
    log_timescale_increment = (math.log(float(max_) / float(min_)) /
      (num*1.0 - 1))

    timescales = min_ * np.exp(
        np.arange(num).astype(float) * log_timescale_increment)
    return timescales

def make_freq_series(min_freqXY, max_freqXY, frequency_num, mid_freqXY = None, freq_init = "geometric"):
    if freq_init == "geometric":
        return compute_geoemtric_series(min_freqXY, max_freqXY, frequency_num)
    elif freq_init == "arith_geometric":
        assert mid_freqXY is not None
        assert min_freqXY < mid_freqXY < max_freqXY
        left_freq_num = int(frequency_num/2)
        right_freq_num = int(frequency_num - left_freq_num)
        
        # left: arithmatric
        left_freqs = np.linspace(start = min_freqXY, stop=mid_freqXY, num=left_freq_num, endpoint=False)
        
        # right: geometric
        right_freqs = compute_geoemtric_series(min_ = mid_freqXY, max_ = max_freqXY, num = right_freq_num)
        
        freqs = np.concatenate([left_freqs, right_freqs], axis = -1)
        return freqs
    elif freq_init == "geometric_arith":
        assert mid_freqXY is not None
        assert min_freqXY < mid_freqXY < max_freqXY
        left_freq_num = int(frequency_num/2)
        right_freq_num = int(frequency_num - left_freq_num)
        
        # left: geometric
        left_freqs = compute_geoemtric_series(min_ = min_freqXY, max_ = mid_freqXY, num = left_freq_num)
        
        # right: arithmatric
        right_freqs = np.linspace(start = mid_freqXY, stop=max_freqXY, num=right_freq_num+1, endpoint=True)
        
        
        
        freqs = np.concatenate([left_freqs, right_freqs[1:]], axis = -1)
        return freqs
    else:
        raise Exception(f"freq_init = {freq_init} is not implemented" )
        
def make_fourier_freq_vector(res_dim, min_freqXY, max_freqXY, mid_freqXY = None, freq_init = "geometric", get_full_vector = True):
    '''
    make the frequency vector for X or Y dimention
    Args:
        res_dim: the total frequency we want
        max_freqXY: the maximum frequency
        min_freqXY: the minimum frequency
        get_full_vector: get the full frequency vector, or half of them (Y dimention)
    '''
    if freq_init == "fft":
        if get_full_vector:
            freq = np.fft.fftfreq(res_dim, d=1/res_dim)
        else:
            freq = np.fft.rfftfreq(res_dim, d=1/res_dim)
    else:
        half_freqs = make_freq_series(min_freqXY, max_freqXY, frequency_num = res_dim//2, 
                                    mid_freqXY = mid_freqXY, freq_init = freq_init)

        if get_full_vector:
            neg_half_freqs = -np.flip(half_freqs, axis = -1)
            if res_dim % 2 == 0:
                freq = np.concatenate([np.array([0.0]), half_freqs[:-1], neg_half_freqs], axis = -1)
            else:
                freq = np.concatenate([np.array([0.0]), half_freqs, neg_half_freqs], axis = -1)
        else:
            if res_dim % 2 == 0:
                freq = np.concatenate([np.array([0.0]), half_freqs], axis = -1)
            else:
                freq = np.concatenate([np.array([0.0]), half_freqs], axis = -1)
    
    if get_full_vector:
        assert freq.shape[0] == res_dim
    else:
        assert freq.shape[0] == math.floor(res_dim*1.0/2)+1
    return freq

def fftfreqs(res, dtype=torch.float32, exact=True, eps = 1e-4):
    """
    Helper function to return frequency tensors
    :param res: n_dims int tuple of number of frequency modes, (fx, fy) for polygon
    :return: omega: 
                if extract  = True
                    for res = (fx, fy) => shape (fx, fy//2+1, 2)
                    for res = (f1, f2, ..., fn) => shape (f1, f2, ..., f_{n-1}, fn//2+1, n)
                if extract  = False
                    for res = (fx, fy) => shape (fx, fy//2, 2)
                    for res = (f1, f2, ..., fn) => shape (f1, f2, ..., f_{n-1}, fn//2, n)
    """

    n_dims = len(res)
    freqs = []
    for dim in range(n_dims - 1):
        r_ = res[dim]
        freq = np.fft.fftfreq(r_, d=1/r_)
        freqs.append(torch.tensor(freq, dtype=dtype))
    r_ = res[-1]
    if exact:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_), dtype=dtype))
    else:
        freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_)[:-1], dtype=dtype))
    omega = torch.meshgrid(freqs)
    omega = list(omega)
    omega = torch.stack(omega, dim=-1)

    omega[0, 0, :] = torch.FloatTensor(np.random.rand(2)*eps).to(omega.dtype)

    return omega

# def fftfreqs(res, dtype=torch.float32, exact=True):
#     """
#     Helper function to return frequency tensors
#     :param res: n_dims int tuple of number of frequency modes, (fx, fy) for polygon
#     :return: omega: 
#                 if extract  = True
#                     for res = (fx, fy) => shape (fx, fy//2+1, 2)
#                     for res = (f1, f2, ..., fn) => shape (f1, f2, ..., f_{n-1}, fn//2+1, n)
#                 if extract  = False
#                     for res = (fx, fy) => shape (fx, fy//2, 2)
#                     for res = (f1, f2, ..., fn) => shape (f1, f2, ..., f_{n-1}, fn//2, n)
#     """

#     n_dims = len(res)
#     freqs = []
#     for dim in range(n_dims - 1):
#         r_ = res[dim]
#         freq = np.fft.fftfreq(r_, d=1/r_)
#         freqs.append(torch.tensor(freq, dtype=dtype))
#     r_ = res[-1]
#     if exact:
#         freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_), dtype=dtype))
#     else:
#         freqs.append(torch.tensor(np.fft.rfftfreq(r_, d=1/r_)[:-1], dtype=dtype))
#     omega = torch.meshgrid(freqs)
#     omega = list(omega)
#     omega = torch.stack(omega, dim=-1)

#     return omega


def permute_seq(i, len):
    """
    Permute the ordering of integer sequences
    """
    assert(i<len)
    return list(range(i, len)) + list(range(0, i))


def img(x, deg=1): # imaginary of tensor (assume last dim: real/imag)
    """
    multiply tensor x by i ** deg
    """
    deg %= 4
    if deg == 0:
        res = x
    elif deg == 1:
        res = x[..., [1, 0]]
        res[..., 0] = -res[..., 0]
    elif deg == 2:
        res = -x
    elif deg == 3:
        res = x[..., [1, 0]]
        res[..., 1] = -res[..., 1]
    return res


def coord_P_lookup_from_V_E(V, E):
    """
    Do polygonal simplex coordinate lookup
    Args:
        V: vertex tensor. float tensor of shape (batch_size, n_vertex, n_dims)
            V: shape (batch_size, n_vertex = num_vert + 1, 2)
        E: element tensor. int tensor of shape (batch_size, n_elem, j+1)
            E: shape (batch_size, n_elem = num_vert, j+1 = 3)
            each value is the index of vertex in V's 2nd dimention
    Return: 
        P: shape (batch_size, n_elem = num_vert, j+1 = 3, n_dims = 2)
        The spatial coordinates of each vertex in E
    """
    assert V.shape[0] == E.shape[0] # batch_size
    assert V.device == E.device
    # batch_idxs: shape (batch_size, 1, 1)
    batch_idxs = torch.arange(0, E.shape[0]).unsqueeze(1).unsqueeze(1)
    # batch_idxs: shape (batch_size, n_elem = num_vert, j+1 = 3)
    batch_idxs = batch_idxs.repeat_interleave(repeats = E.shape[1], dim=1).repeat_interleave(repeats = E.shape[2], dim=2)
    batch_idxs = batch_idxs.to(V.device)
    # P: shape (batch_size, n_elem = num_vert, j+1 = 3, n_dims = 2)
    P = V[batch_idxs, E]
    return P


def construct_B(V, E):
    """
    Construct B matrix for Cayley-Menger Determinant
    Args:
        V: vertex tensor. float tensor of shape (batch_size, n_vertex, n_dims)
            V: shape (batch_size, n_vertex = num_vert + 1, 2)
        E: element tensor. int tensor of shape (batch_size, n_elem, j+1)
            E: shape (batch_size, n_elem = num_vert, j+1 = 3)
            each value is the index of vertex in V's 2nd dimention
    Return:
        B: B matrix of shape (batch_size, n_elem, j+2, j+2)

    compute the Cayley-Menger Determinant exactly as:
    1. Equation 5 in https://openreview.net/pdf?id=B1G5ViAqFm
    2. Equation 6 in https://arxiv.org/pdf/1901.11082.pdf
    """
    assert V.shape[0] == E.shape[0] # batch_size
    batch_size = V.shape[0]
    # ne: n_elem, number of lines in polygon boundary
    ne = E.shape[-2]
    # j = 2
    j = E.shape[-1]-1
    # P: point tensor. float, shape (batch_size, n_elem = num_vert, j+1 = 3, n_dims = 2)
    # P = V[E]
    P = coord_P_lookup_from_V_E(V, E)
    
    B = torch.zeros(batch_size, ne, j+2, j+2, device=V.device, dtype=V.dtype)
    B[:, :, :, 0] = 1
    B[:, :, 0, :] = 1
    # 2-for-loop only loop 3 times for j = 2
    for r in range(1, j+2):
        for c in range(r+1, j+2):
            B[:, :, r, c] = torch.sum((P[:, :, r-1] - P[:, :, c-1]) ** 2, dim=-1)
            B[:, :, c, r] = B[:, :, r, c]
    B[:, :, 0, 0] = 0
    
    return B

def simplex_content(V, E, signed=False):
    """
    Compute the content of simplices in a simplicial complex
    This essentailly compute the C_n^j for a simplex in Equation 6
    :param V: vertex tensor. float tensor of shape (batch_size, n_vertex, n_dims)
            V: shape (batch_size, n_vertex = num_vert + 1, 2)
    :param E: element tensor. int tensor of shape (batch_size, n_elem, j+1)
            E: shape (batch_size, n_elem = num_vert, j+1 = 3)
    :param signed: bool denoting whether to calculate signed content or unsigned content
            True:  the polygon case
            False: for other case
    :return: vol: volume of the simplex, shape (batch_size, n_elem = num_vert, 1)
    """
    # ne: n_elem, number of lines in polygon boundary
    ne = E.shape[-2]
    # nd = 2
    nd = V.shape[-1]
    # j = 2
    j = E.shape[-1]-1
    # P: shape (batch_size, n_elem = num_vert, j+1 = 3, n_dims = 2)
    # P = V[E]
    P = coord_P_lookup_from_V_E(V, E)

    if not signed:
        # B: B matrix of shape (batch_size, n_elem, j+2, j+2)
        B = construct_B(V, E) # construct Cayley-Menger matrix
        # vol2: the square of the content of each 2-simplex, shape (batch_size, n_elem)
        #       Equation 5 in https://openreview.net/pdf?id=B1G5ViAqFm
        # vol2 = (-1)**(j+1) / (2**j) / (factorial(j)**2) * batch_det(B)
        vol2 = (-1)**(j+1) / (2**j) / (math.factorial(j)**2) * torch.linalg.det(B)
        # neg_mask: shape (batch_size), the index of the negative simplex????
        neg_mask = torch.sum(vol2 < 0, dim = -1)
        if torch.sum(neg_mask) > 0:
            vol2[:, neg_mask] = 0
            print("[!]Warning: zeroing {0} small negative number".format(torch.sum(neg_mask).item()))
        vol =torch.sqrt(vol2)
    else:
        # Equation 8 in https://openreview.net/pdf?id=B1G5ViAqFm
        # compute based on: https://en.m.wikipedia.org/wiki/Simplex#Volume
        assert(j == nd)
        # matrix determinant
        # x_{j+1} is auxilarity node with (0, 0) coordinate
        # [x1,x2,...,x_{j}] - [x_{j+1}]
        # P[:, :, :-1]: [x1,x2,...,x_{j}], shape (batch_size, n_elem = num_vert, j = 2, n_dims = 2), 
        # P[:, :, -1:]: [x_{j+1}], shape (batch_size, n_elem = num_vert, 1, n_dims = 2)
        # mat: shape (batch_size, n_elem = num_vert, j = 2, n_dims = 2)
        mat = P[:, :, :-1] - P[:, :, -1:]
        # vol: shape (batch_size, n_elem = num_vert)
        vol = torch.linalg.det(mat) / math.factorial(j)

    return vol.unsqueeze(-1)


def batch_det(A):
    """
    (No use) Batch compute determinant of square matrix A of shape (*, N, N)

    We can use torch.linalg.det() directly
    Return:
    Tensor of shape (*)

    第一种初等变换—bai—交换两行或du列—zhi—要偶数次不改变dao行列的值
    第二种初等zhuan变换——shu某行（列）乘以非0实数——这个可以乘以系数，但总的乘积必须为1方可不改变行列式值
    第三种初等变换——某行（列）乘以实数加到另一行（列）上——此条对行列式值无影响
    """
    # 对矩阵进行LU 分解，显然，分解后的矩阵对角线上元素的乘积即为原始矩阵行列式的值
    # 由于在进行 LU 分解时可能会进行行交换的情况，而每次行交换都会带来行列式的符号变化，所以我们要记录行交换次数的奇偶性，
    # 奇数则符号改变，偶数则符号不变，请密切注意下面程序中出现的用来标记变换次数奇偶性的变量 parity 
    # LU: the packed LU factorization matrix, (*, N, N)
    # pivots: (*, N)
    LU, pivots = torch.lu(A)
    # torch.einsum('...ii->...i', LU): the diagnoal vector
    # det_LU: the product of all diagnal values
    det_LU = torch.einsum('...ii->...i', LU).prod(-1)
    pivots -= 1
    d = pivots.shape[-1]
    perm = pivots - torch.arange(d, dtype=pivots.dtype, device=pivots.device).expand(pivots.shape)
    det_P = (-1) ** ((perm != 0).sum(-1))
    det = det_LU * det_P.type(det_LU.dtype)

    return det



def make_E(V):
    '''
    here, we assume V comes from a simple polygon
    Given polygon vertice tensor -> V with shape (batch_size, num_vert, 2)
    Generate its edge matrix E
    Note, num_vert reflect all unique vertices, remove the repeated last/first node beforehand
    Here, num_vert: number of vertice of input polygon = number of edges = number pf 2-simplex (auxiliary node)
    
    Args:
        V with shape (batch_size, num_vert, 2)
    Return:
        E: torch.LongTensor(), shape (batch_size, num_vert, 2)
    '''
    batch_size, num_vert, n_dims = V.shape
    a = torch.arange(0, num_vert)
    E = torch.stack((a, a+1), dim = 0).permute(1,0)
    E[-1, -1] = 0
    E = torch.repeat_interleave(E.unsqueeze(0), repeats = batch_size, dim=0).to(V.device)
    return E

def make_D(V):
    '''
    Given polygon vertice tensor -> V with shape (batch_size, num_vert, 2)
    Generate its density matrix D
    Note, num_vert reflect all unique vertices, remove the repeated last/first node beforehand
    Here, num_vert: number of vertice of input polygon = number of edges = number pf 2-simplex (auxiliary node)
    
    Args:
        V: shape (batch_size, num_vert, 2)
    Return:
        D: torch.LongTensor(), shape (batch_size, num_vert, 1)
    '''
    batch_size, num_vert, n_dims = V.shape
    D = torch.ones(batch_size, num_vert, 1).to(V.device)
    return D

def affinity_V(V, extent):
    '''
    affinity vertice tensor to move it to [0, periodX, 0, periodY]
    
    Args:
        V: torch.FloatTensor(), shape (batch_size, num_vert, n_dims = 2)
            vertex tensor
        extent: the maximum spatial extent of all polygons, (minx, maxx, miny, maxy)
        
        eps: the maximum noise we add to each polygon vertice
    Retunr:
        V: torch.FloatTensor(), shape (batch_size, num_vert, n_dims = 2)
    '''
    device = V.device
    minx, maxx, miny, maxy = extent

    # assert maxx - minx == periodXY[0]
    # assert maxy - miny == periodXY[1]

    # affinity all polygons to make them has positive coordinates
    # move to (0,2,0,2)
    V = V + torch.FloatTensor([-minx, -miny]).to(device)
    
    return V


def add_noise_V(V, eps):
    '''
    add small noise to each vertice to make NUFT more robust
    Args:
        V: torch.FloatTensor(), shape (batch_size, num_vert, n_dims = 2)
            vertex tensor
        eps: the maximum noise we add to each polygon vertice
    Retunr:
        V: torch.FloatTensor(), shape (batch_size, num_vert, n_dims = 2)
    '''
    # add small noise
    V = V + torch.rand(V.shape, device = V.device)*eps
    return V






def make_periodXY(extent):
    '''
    Make periodXY based on the spatial extent
    Args:
        extent: (minx, maxx, miny, maxy)
    Return:
        periodXY: t in DDSL_spec(), [periodX, periodY]
            periodX, periodY: the spatial extend from [0, periodX]
    '''
    minx, maxx, miny, maxy = extent

    periodX =  maxx - minx 
    periodY =  maxy - miny

    periodXY = [periodX, periodY]
    return periodXY


def polygon_nuft_input(polygons, extent, V = None, E = None):
    '''
    polygons: torch.FloatTensor(), shape (batch_size, num_vert, n_dims = 2)
        last points not equal to the 1st one
    extent: the maximum spatial extent of all polygons, (minx, maxx, miny, maxy)
    '''
    assert (polygons is None and V is not None and E is not None) or (polygons is not None and V is None and E is None)
    if polygons is not None:
        # V: torch.FloatTensor(), shape (batch_size, num_vert, n_dims = 2)
        #    vertex tensor 
        V = polygons
    # affinity vertice tensor to move it to [0, periodX, 0, periodY]
    V = affinity_V(V, extent)
    if E is None:
        # add noise
        # V = add_noise_V(V, self.eps)
        # E: torch.LongTensor(),  shape (batch_size, num_vert, 2) 
        #    element tensor, each element [[0,1], [1,2],...,[num_vert-1, 0]]
        E = make_E(V)
    # D: torch.LongTensor(),  shape (batch_size, num_vert, 1)
    #    all be one tensor
    D = make_D(V)
    
    
    return V, E, D