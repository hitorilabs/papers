# TAKEN FROM @BIRCHLABS
# https://github.com/Birch-san/booru-embed

import torch
from torch.utils.flop_counter import FlopCounterMode
from triton.testing import do_bench
from torch import FloatTensor, Size
from functorch.einops import rearrange
from torch.nn import Module, Linear
from torch.nn.functional import scaled_dot_product_attention
from xformers.ops import memory_efficient_attention, MemoryEfficientAttentionCutlassOp
from typing import Optional, Literal, NamedTuple

class Attn(Module):
  q_proj: Linear
  k_proj: Linear
  v_proj: Linear
  heads: int
  use_xformers: bool
  def __init__(self, in_dim=320, head_dim=64, heads=8, use_xformers=False) -> None:
    super().__init__()
    self.heads = heads
    # yes I know you can fuse them
    self.q_proj = Linear(in_features=in_dim, out_features=heads*head_dim)
    self.k_proj = Linear(in_features=in_dim, out_features=heads*head_dim)
    self.v_proj = Linear(in_features=in_dim, out_features=heads*head_dim)
    self.use_xformers = use_xformers
  
  def forward(self, x: FloatTensor, bias: FloatTensor) -> FloatTensor:
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    rearr = 'b t (h c) -> b t h c' if self.use_xformers else 'b t (h c) -> b h t c'
    q = rearrange(q, rearr, h=self.heads)
    k = rearrange(k, rearr, h=self.heads)
    v = rearrange(v, rearr, h=self.heads)

    bias = rearrange(bias, 'b t -> b 1 t 1')
    if self.use_xformers:
      # b h t t
      bias = bias.expand(-1, self.heads, -1, k.size(1)).contiguous()
      # b l h c
      scores = memory_efficient_attention(q, k, v, attn_bias=bias)
    else:
      # b h l c
      scores = scaled_dot_product_attention(q, k, v, attn_mask=bias)
    return scores

cutlass_fwd, cutlass_bwd = MemoryEfficientAttentionCutlassOp

class TensorMeta:
  shape: Size
  ndim: int
  def __init__(self, shape: Size) -> None:
    self.shape = shape
    self.ndim = len(shape)

class CutlassFwdOutShape(NamedTuple):
  # b t h c
  attn_scores: Size
  # b h t
  logsumexp: Size
  rng_seed: int
  rng_offset: int

def cutlass_fwd_flop(
  query: Size,
  key: Size,
  value: Size,
  attn_bias: Optional[Size],
  seqstart_q: Optional[Size],
  seqstart_k: Optional[Size],
  max_seqlen_q: int,
  dropout_p: float,
  compute_logsumexp: bool,
  custom_mask_type: Literal[0, 1, 2],
  scale: Optional[float],
  seqlen_k: Optional[bool],
  window_size: int,
  out_shape: CutlassFwdOutShape = None,
  **kwargs,
):
  assert seqstart_q is None and seqstart_k is None, "Cannot compute flops due to use of BlockDiagonalMask/BlockDiagonalCausalWithOffsetPaddedKeysMask. we need the tensor information contained in seqstart_q and seqstart_k, but FlopCounter's torch_dispatch only gave us the shapes, not the data."
  # this thing expects to receive tensors, but we don't have any
  # fortunately it's only interested in their shapes and dims
  return cutlass_fwd.operator_flop(
    TensorMeta(query),
    TensorMeta(key),
    TensorMeta(value),
    attn_bias, # unused
    seqstart_q,
    seqstart_k,
    max_seqlen_q, # unused
    compute_logsumexp, # unused
    custom_mask_type,
  )

class CutlassBwdOutShape(NamedTuple):
  # b t h c
  grad_query: Size
  grad_key: Size
  grad_value: Size
  # shouldn't require grad, so None (except perhaps if you made a mistake)
  grad_bias: Optional[Size]

def cutlass_bwd_flop(
  grad: Size,
  query: Size,
  key: Size,
  value: Size,
  attn_bias: Optional[Size],
  cu_seqlens_q: Optional[Size],
  cu_seqlens_k: Optional[Size],
  max_seqlen_q: int,
  max_seqlen_k: int,
  logsumexp: Size,
  output: Size,
  dropout_p: float,
  rng_seed: int,
  rng_offset: int,
  custom_mask_type: Literal[0, 1, 2],
  scale: Optional[float],
  num_splits_key: int,
  window_size: int,
  out_shape: CutlassBwdOutShape = None,
):
  assert cu_seqlens_q is None and cu_seqlens_k is None, "Cannot compute flops due to use of BlockDiagonalMask/BlockDiagonalCausalWithOffsetPaddedKeysMask. we need the tensor information contained in seqstart_q and seqstart_k, but FlopCounter's torch_dispatch only gave us the shapes, not the data."
  # this thing expects to receive tensors, but we don't have any
  # fortunately it's only interested in their shapes and dims
  return cutlass_bwd.operator_flop(
    grad, # unused
    TensorMeta(query),
    TensorMeta(key),
    TensorMeta(value),
    attn_bias, # unused
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q, # unused
    max_seqlen_k, # unused
    logsumexp, # unused
    output, # unused
    dropout_p, # unused
    rng_seed, # unused
    rng_offset, # unused
    custom_mask_type,
    scale, # unused
  )

def get_flops_achieved(f):
  flop_counter = FlopCounterMode(display=True, custom_mapping={
    cutlass_fwd.OPERATOR: cutlass_fwd_flop,
    cutlass_bwd.OPERATOR: cutlass_bwd_flop,
  })
  with flop_counter:
    f()
  total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
  ms_per_iter = do_bench(f)
  iters_per_second = 1e3/ms_per_iter
  print(f"{iters_per_second * total_flops / 1e12} TF/s")

device=torch.device('cuda')
dtype=torch.bfloat16

in_dim=320
sdp_model = Attn(in_dim=in_dim, head_dim=64, heads=8, use_xformers=False).to(device=device, dtype=dtype)
xfo_model = Attn(in_dim=in_dim, head_dim=64, heads=8, use_xformers=True).to(device=device, dtype=dtype)
inp = torch.randn(8, 4096, in_dim, device=device, dtype=dtype)
bias = torch.randn(8, 4096, device=device, dtype=dtype)

print('tracing sdp...')
get_flops_achieved(lambda: sdp_model(inp, bias).sum().backward())
print('tracing xformers...')
get_flops_achieved(lambda: xfo_model(inp, bias).sum().backward())

print('tracing compiled sdp...')
sdp_compiled = torch.compile(sdp_model)
get_flops_achieved(lambda: sdp_compiled(inp, bias).sum().backward())
print('tracing compiled xformers...')
xfo_compiled = torch.compile(xfo_model)
get_flops_achieved(lambda: xfo_compiled(inp, bias).sum().backward())
