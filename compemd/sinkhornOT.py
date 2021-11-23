import math
import os.path

import numpy as np
import torch
import torch.utils.cpp_extension

from settings import DEVICE

"""
taken from
https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/Pytorch_Wasserstein.ipynb
"""

# scales used to build distance matrix
# https://iopscience.iop.org/article/10.1088/1367-2630/18/9/093011/pdf
ModifiedPettiforScale = ("He", "Ne", "Ar", "Kr", "Xe", "Rn", "Fr", "Cs", "Rb", "K", "Na", "Li", "Ra", "Ba", "Sr", "Ca",
                         "Eu", "Yb", "Lu", "Tm", "Y", "Er", "Ho", "Dy", "Tb", "Gd", "Sm", "Pm", "Nd", "Pr", "Ce", "La",
                         "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Sc",
                         "Zr", "Hf", "Ti", "Ta", "Nb", "V", "Cr", "Mo", "W", "Re", "Tc", "Os", "Ru", "Ir", "Rh", "Pt",
                         "Pd", "Au", "Ag", "Cu", "Ni", "Co", "Fe", "Mn", "Mg", "Zn", "Cd", "Hf", "Be", "Al", "Ga", "In",
                         "Tl", "Pb", "Sn", "Ge", "Si", "B", "C", "N", "P", "As", "Sb", "Bi", "Po", "Te", "Se", "S", "O",
                         "At", "I", "Br", "Cl", "F", "H")
# https://dx.doi.org/10.1021/acs.jpcc.0c07857?ref=pdf
UniversalSequenceScale = ("Fr", "Cs", "Rb", "K", "Ra", "Ba", "Sr", "Ac", "Ca", "Na", "Rn", "Yb", "La", "Pm", "Tb", "Sm",
                          "Gd", "Eu", "Y", "Dy", "Th", "Ho", "Er", "Tm", "Lu", "Li", "Ce", "Mg", "Pr", "Hf", "Xe", "Zr",
                          "Nd", "Sc", "Tl", "Pa", "Pu", "U", "Cm", "Am", "Np", "Cd", "Pb", "Ta", "In", "Po", "At", "Nb",
                          "Ti", "Al", "Bi", "Sn", "Zn", "Hg", "Te", "Sb", "Ga", "V", "Mn", "Ag", "Cr", "Be", "Kr", "Ge",
                          "Re", "Si", "Tc", "Cu", "I", "Fe", "As", "Ni", "Co", "Mo", "Ar", "Pd", "Ir", "Os", "Pt", "Ru",
                          "P", "Rh", "W", "Se", "Au", "B", "S", "Br", "Cl", "H", "Ne", "He", "C", "N", "O", "F")

MendeleevScale = ("H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                  "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
                  "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
                  "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
                  "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
                  "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm")

cuda_source = """

#include <torch/extension.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>

using at::RestrictPtrTraits;
using at::PackedTensorAccessor;

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
#endif

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}


template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

// While this might be the most efficient sinkhorn step / logsumexp-matmul implementation I have seen,
// this is awfully inefficient compared to matrix multiplication and e.g. NVidia cutlass may provide
// many great ideas for improvement
template <typename scalar_t, typename index_t>
__global__ void sinkstep_kernel(
  // compute log v_bj = log nu_bj - logsumexp_i 1/lambda dist_ij - log u_bi
  // for this compute maxdiff_bj = max_i(1/lambda dist_ij - log u_bi)
  // i = reduction dim, using threadIdx.x
  PackedTensorAccessor<scalar_t, 2, RestrictPtrTraits, index_t> log_v,
  const PackedTensorAccessor<scalar_t, 2, RestrictPtrTraits, index_t> dist,
  const PackedTensorAccessor<scalar_t, 2, RestrictPtrTraits, index_t> log_nu,
  const PackedTensorAccessor<scalar_t, 2, RestrictPtrTraits, index_t> log_u,
  const scalar_t lambda) {

  using accscalar_t = scalar_t;

  __shared__ accscalar_t shared_mem[2 * WARP_SIZE];

  index_t b = blockIdx.y;
  index_t j = blockIdx.x;
  int tid = threadIdx.x;

  if (b >= log_u.size(0) || j >= log_v.size(1)) {
    return;
  }
  // reduce within thread
  accscalar_t max = -std::numeric_limits<accscalar_t>::infinity();
  accscalar_t sumexp = 0;

  if (log_nu[b][j] == -std::numeric_limits<accscalar_t>::infinity()) {
    if (tid == 0) {
      log_v[b][j] = -std::numeric_limits<accscalar_t>::infinity();
    }
    return;
  }

  for (index_t i = threadIdx.x; i < log_u.size(1); i += blockDim.x) {
    accscalar_t oldmax = max;
    accscalar_t value = -dist[i][j]/lambda + log_u[b][i];
    max = max > value ? max : value;
    if (oldmax == -std::numeric_limits<accscalar_t>::infinity()) {
      // sumexp used to be 0, so the new max is value and we can set 1 here,
      // because we will come back here again
      sumexp = 1;
    } else {
      sumexp *= exp(oldmax - max);
      sumexp += exp(value - max); // if oldmax was not -infinity, max is not either...
    }
  }

  // now we have one value per thread. we'll make it into one value per warp
  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    accscalar_t o_max    = WARP_SHFL_XOR(max, 1 << i, WARP_SIZE);
    accscalar_t o_sumexp = WARP_SHFL_XOR(sumexp, 1 << i, WARP_SIZE);
    if (o_max > max) { // we're less concerned about divergence here
      sumexp *= exp(max - o_max);
      sumexp += o_sumexp;
      max = o_max;
    } else if (max != -std::numeric_limits<accscalar_t>::infinity()) {
      sumexp += o_sumexp * exp(o_max - max);
    }
  }

  __syncthreads();
  // this writes each warps accumulation into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  if (tid % WARP_SIZE == 0) {
    shared_mem[tid / WARP_SIZE * 2] = max;
    shared_mem[tid / WARP_SIZE * 2 + 1] = sumexp;
  }
  __syncthreads();
  if (tid < WARP_SIZE) {
    max = (tid < blockDim.x / WARP_SIZE ? shared_mem[2 * tid] : -std::numeric_limits<accscalar_t>::infinity());
    sumexp = (tid < blockDim.x / WARP_SIZE ? shared_mem[2 * tid + 1] : 0);
  }
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    accscalar_t o_max    = WARP_SHFL_XOR(max, 1 << i, WARP_SIZE);
    accscalar_t o_sumexp = WARP_SHFL_XOR(sumexp, 1 << i, WARP_SIZE);
    if (o_max > max) { // we're less concerned about divergence here
      sumexp *= exp(max - o_max);
      sumexp += o_sumexp;
      max = o_max;
    } else if (max != -std::numeric_limits<accscalar_t>::infinity()) {
      sumexp += o_sumexp * exp(o_max - max);
    }
  }

  if (tid == 0) {
    log_v[b][j] = (max > -std::numeric_limits<accscalar_t>::infinity() ?
                   log_nu[b][j] - log(sumexp) - max : 
                   -std::numeric_limits<accscalar_t>::infinity());
  }
}

template <typename scalar_t>
torch::Tensor sinkstep_cuda_template(const torch::Tensor& dist, const torch::Tensor& log_nu, const torch::Tensor& log_u,
                                     const double lambda) {
  TORCH_CHECK(dist.is_cuda(), "need cuda tensors");
  TORCH_CHECK(dist.device() == log_nu.device() && dist.device() == log_u.device(), "need tensors on same GPU");
  TORCH_CHECK(dist.dim()==2 && log_nu.dim()==2 && log_u.dim()==2, "invalid sizes");
  TORCH_CHECK(dist.size(0) == log_u.size(1) &&
           dist.size(1) == log_nu.size(1) &&
           log_u.size(0) == log_nu.size(0), "invalid sizes");
  auto log_v = torch::empty_like(log_nu);
  using index_t = int32_t;

  auto log_v_a = log_v.packed_accessor<scalar_t, 2, RestrictPtrTraits, index_t>();
  auto dist_a = dist.packed_accessor<scalar_t, 2, RestrictPtrTraits, index_t>();
  auto log_nu_a = log_nu.packed_accessor<scalar_t, 2, RestrictPtrTraits, index_t>();
  auto log_u_a = log_u.packed_accessor<scalar_t, 2, RestrictPtrTraits, index_t>();

  auto stream = at::cuda::getCurrentCUDAStream();

  int tf = getNumThreads(log_u.size(1));
  dim3 blocks(log_v.size(1), log_u.size(0));
  dim3 threads(tf);

  sinkstep_kernel<<<blocks, threads, 2*WARP_SIZE*sizeof(scalar_t), stream>>>(
    log_v_a, dist_a, log_nu_a, log_u_a, static_cast<scalar_t>(lambda)
    );

  return log_v;
}

torch::Tensor sinkstep_cuda(const torch::Tensor& dist, const torch::Tensor& log_nu, const torch::Tensor& log_u,
                            const double lambda) {
    return AT_DISPATCH_FLOATING_TYPES(log_u.scalar_type(), "sinkstep", [&] {
       return sinkstep_cuda_template<scalar_t>(dist, log_nu, log_u, lambda);
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sinkstep", &sinkstep_cuda, "sinkhorn step");
}

"""

build_target = torch.utils.cpp_extension.get_default_build_root()
build_target = os.path.join(build_target, "wasserstein")

try:
    wasserstein_ext = torch.utils.cpp_extension._import_module_from_library("wasserstein",
                                                                            build_target,
                                                                            True)
except:
    wasserstein_ext = torch.utils.cpp_extension.load_inline("wasserstein", cpp_sources="", cuda_sources=cuda_source,
                                                            extra_cuda_cflags=["--expt-relaxed-constexpr"])


def sinkstep(dist, log_nu, log_u, lam: float):
    # dispatch to optimized GPU implementation for GPU tensors, slow fallback for CPU
    if dist.is_cuda:
        return wasserstein_ext.sinkstep(dist, log_nu, log_u, lam)
    assert dist.dim() == 2 and log_nu.dim() == 2 and log_u.dim() == 2
    assert dist.size(0) == log_u.size(1) and dist.size(1) == log_nu.size(1) and log_u.size(0) == log_nu.size(0)
    log_v = log_nu.clone()
    for b in range(log_u.size(0)):
        log_v[b] -= torch.logsumexp(-dist / lam + log_u[b, :, None], 0)
    return log_v


class SinkhornOT(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mu, nu, dist, lam=1e-1, N=2000):
        assert mu.dim() == 2 and nu.dim() == 2 and dist.dim() == 2
        bs = mu.size(0)
        d1, d2 = dist.size()
        assert nu.size(0) == bs and mu.size(1) == d1 and nu.size(1) == d2
        log_mu = mu.log()
        log_nu = nu.log()
        log_u = torch.full_like(mu, -math.log(d1))
        log_v = torch.full_like(nu, -math.log(d2))
        for i in range(N):
            log_v = sinkstep(dist, log_nu, log_u, lam)
            log_u = sinkstep(dist.t(), log_mu, log_v, lam)

        # this is slight abuse of the function. it computes (diag(exp(log_u))*Mt*exp(-Mt/lam)*diag(exp(log_v))).sum()
        # in an efficient (i.e. no bxnxm tensors) way in log space
        distances = (-sinkstep(-dist.log() + dist / lam, -log_v, log_u, 1.0)).logsumexp(1).exp()
        ctx.log_v = log_v
        ctx.log_u = log_u
        ctx.dist = dist
        ctx.lam = lam
        return distances

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out[:, None] * ctx.log_u * ctx.lam, grad_out[:, None] * ctx.log_v * ctx.lam, None, None, None


def emdloss(mu, nu, dist, N=5000, lam=1e-1):
    """loss function used in gan"""
    assert mu.dim() == 2 and nu.dim() == 2 and dist.dim() == 2
    bs = mu.size(0)
    d1, d2 = dist.size()
    assert nu.size(0) == bs and mu.size(1) == d1 and nu.size(1) == d2
    log_mu = mu.log()
    log_nu = nu.log()
    log_u = torch.full_like(mu, -math.log(d1))
    log_v = torch.full_like(nu, -math.log(d2))
    for i in range(N):
        log_v = sinkstep(dist, log_nu, log_u, lam)
        log_u = sinkstep(dist.t(), log_mu, log_v, lam)

    # this is slight abuse of the function. it computes (diag(exp(log_u))*Mt*exp(-Mt/lam)*diag(exp(log_v))).sum()
    # in an efficient (i.e. no bxnxm tensors) way in log space
    distances = (-sinkstep(-dist.log() + dist / lam, -log_v, log_u, 1.0)).logsumexp(1).exp()
    return distances.mean()


def emdloss_detach(ta, tb, distance_matrix):
    ta = ta.detach()
    tb = tb.detach()
    return SinkhornOT.apply(ta, tb, distance_matrix, 1e-1, 5000)


def element_distance_matrix(possible_elements: [str], sequence_used: [str], return_tensor=True):
    """
    return distance mat for an element sequence
    """
    size = len(possible_elements)
    m = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        ei = possible_elements[i]
        for j in range(i, size):
            ej = possible_elements[j]
            dij = abs(sequence_used.index(ei) - sequence_used.index(ej))
            m[i][j] = dij
            m[j][i] = dij
    if return_tensor:
        return torch.tensor(m, dtype=torch.float)
    else:
        return m


def dummy_distance_matrix(ndim: int, interval=3):
    d = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            d[i][j] = abs(i - j) * float(interval)
    return d


def chememd_distance_matrix(encoded, distmat, chunk=2000):
    """given a list of normalized points and a distance matrix, return the emd distance matrix of these points """
    if not torch.is_tensor(distmat):
        distmat = torch.tensor(distmat)
    if torch.is_tensor(encoded):
        encoded = encoded.detach().cpu().numpy()
    distmat = distmat.to(DEVICE, dtype=torch.float32)
    # get the pairs for the triangle of dist mat
    ijs = []
    encoded_is = []
    encoded_js = []
    for i in range(len(encoded)):
        for j in range(i, len(encoded)):
            encoded_is.append(encoded[i])
            encoded_js.append(encoded[j])
            ijs.append((i, j))
    encoded_is = np.array(encoded_is)
    encoded_js = np.array(encoded_js)
    print("the distance matrix should have # of elements:", len(encoded) * len(encoded))
    print("number of elements in the triangle:", len(ijs))
    print("compute emd distance matrix...")

    # calculate distances using sinkhorn optimal transport
    from tqdm import tqdm
    dists = []
    for i in tqdm(range(len(encoded_is) // chunk + 1)):
        ta = torch.tensor(encoded_is[i * chunk:(i + 1) * chunk], dtype=torch.float).to(DEVICE)
        tb = torch.tensor(encoded_js[i * chunk:(i + 1) * chunk], dtype=torch.float).to(DEVICE)
        emds = emdloss_detach(ta, tb, distmat)
        print("chunk", i, ":", ta.shape, emds.mean())
        dists += emds.tolist()
    emd_distmat = np.zeros((len(encoded), len(encoded)))
    idist = 0
    for ij in ijs:
        i, j = ij
        emd_distmat[i, j] = dists[idist]
        emd_distmat[j, i] = dists[idist]
        idist += 1
    return emd_distmat


def check_against_pyemd():
    from pyemd import emd
    from sklearn.preprocessing import normalize

    a = np.random.random((4, 6))
    b = np.random.random((4, 6))
    a = normalize(a, axis=1, norm="l1")
    b = normalize(b, axis=1, norm="l1")
    d = dummy_distance_matrix(a.shape[1])
    for i in range(a.shape[0]):
        print("pyemd gives:", emd(a[i], b[i], d))

    ta = torch.tensor(a).to(DEVICE, dtype=torch.float32)
    tb = torch.tensor(b).to(DEVICE, dtype=torch.float32)
    td = torch.tensor(d).to(DEVICE, dtype=torch.float32)
    emd_dist = emdloss_detach(ta, tb, td)
    print("sinkhorn gives:", emd_dist)

    # emd_dist_mat = chememd_distance_matrix(ta, td)
    # print("sinkhorn gives:",emd_dist_mat)


if __name__ == '__main__':
    # check_against_pyemd()  # uncomment to verify sinkhorn

    # use the following to generate emd distance matrix
    from data.schema import AtmoStructureGroup, FormulaEncoder
    from utils import save_pkl

    scale_used = ModifiedPettiforScale

    master_group = AtmoStructureGroup.from_csv()
    fe = FormulaEncoder(master_group.possible_elements)
    encoded = fe.encode_2d([s.composition.fractional_composition.formula for s in master_group])
    emd_dist = element_distance_matrix(master_group.possible_elements, scale_used, True)
    chem_emd = chememd_distance_matrix(encoded, emd_dist, 50000)  # exploded for chunk=100k on 2080
    data = {
        "master_group": master_group,
        "emd_distance_matrix": chem_emd,
        "encoded": encoded,
        "scale": scale_used,
    }
    save_pkl(data, "../dataset/emd_data.pkl")
