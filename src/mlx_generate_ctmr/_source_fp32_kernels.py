"""Metal FP32 operation orders used by the pinned NV-Generate image source.

Kernels keep FMA and separately rounded operations explicit. The matrix cases
mirror the observed CUDA dispatch; their qualification bounds are documented in
the source arithmetic manifest. No model activations are embedded here.
"""

from __future__ import annotations

_HEADER_0 = """

#pragma clang fp contract(off)
struct WF {
    float m, v, n;
};
WF combine(WF a, WF b) {
    if (a.n == 0)
        return b;
    if (b.n == 0)
        return a;
    float d = b.m - a.m;
    float n = a.n + b.n;
    float p = b.n / n;
    return {metal::fma(d, p, a.m), metal::fma(d * d * a.n, p, a.v + b.v), n};
}
"""

_HEADER_1 = """

#pragma clang fp contract(off)
float exp_cuda(float x, const device uint *T) {
    if (metal::isnan(x))
        return x;
    if (x > 89)
        return INFINITY;
    if (x < -104)
        return 0;
    float t = metal::clamp(metal::fma(x, as_type<float>(0x3bbb989du), 0.5f), 0.f, 1.f);
    float p = t * 252.f;
    float rem = metal::fma(t, 252.f, -p);
    float nf = metal::floor(p);
    if (p == nf && rem < 0)
        nf -= 1;
    float n = nf - 126.f;
    float a = metal::fma(x, as_type<float>(0x3fb8aa3bu), -n);
    float f = metal::fma(x, as_type<float>(0x32a57060u), a);
    if (f < 0 && f >= -1) {
        uint u = as_type<uint>(-f);
        uint hint = 0x3f800000u - as_type<uint>(metal::precise::exp2(f));
        const device uint *N = T + 8388609;
        while (hint > 0 && u < N[hint])
            hint -= 1;
        while (hint < 8388608u && u >= N[hint + 1])
            hint += 1;
        return metal::ldexp(as_type<float>(0x3f800000u - hint), int(n));
    }
    float ef = metal::floor(f);
    float frac = f - ef;
    uint u = as_type<uint>(frac);
    uint hint = as_type<uint>(metal::precise::exp2(frac)) - 0x3f800000u;
    hint = metal::min(hint, 8388608u);
    while (hint > 0 && u < T[hint])
        hint -= 1;
    while (hint < 8388608u && u >= T[hint + 1])
        hint += 1;
    float e2 = as_type<float>(0x3f800000u + hint);
    return metal::ldexp(e2, int(n + ef));
}
"""

_OPS_CONV = """

#pragma clang fp contract(off)
uint flat = thread_position_in_grid.x;
uint width = X_shape[4], height = X_shape[3], depth = X_shape[2], cin = X_shape[1],
     cout = W_shape[0];
uint ow = (width + 2 * P - KS) / S + 1, oh = (height + 2 * P - KS) / S + 1,
     od = (depth + 2 * P - KS) / S + 1;
uint vol = width * height * depth, ovol = ow * oh * od;
if (flat >= X_shape[0] * cout * ovol)
    return;
int ox = int(flat % ow) * S - P, oy = int((flat / ow) % oh) * S - P,
    oz = int((flat / (ow * oh)) % od) * S - P;
uint co = (flat / ovol) % cout, batch = flat / (ovol * cout);
float acc = 0;
for (uint ci = 0; ci < cin; ++ci) {
    for (uint kz = 0; kz < KS; ++kz) {
        int zz = oz + kz;
        for (uint ky = 0; ky < KS; ++ky) {
            int yy = oy + ky;
            for (uint kx = 0; kx < KS; ++kx) {
                int xx = ox + kx;
                float x = 0;
                if (xx >= 0 && xx < int(width) && yy >= 0 && yy < int(height) && zz >= 0 &&
                    zz < int(depth))
                    x = X[(batch * cin + ci) * vol + uint(zz) * width * height +
                          uint(yy) * width + uint(xx)];
                float w = W[(((co * cin + ci) * KS + kz) * KS + ky) * KS + kx];
                acc = metal::fma(x, w, acc);
            }
        }
    }
}
Y[flat] = acc + B[co];
"""

_OPS_LINEAR = """

#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
uint cin = W_shape[1], cout = W_shape[0];
if (i >= ROWS * cout)
    return;
uint row = i / cout, co = i % cout;
float parts[16];
for (uint k = 0; k < 16; ++k)
    parts[k] = 0;
for (uint start = 0; start < cin; start += 256) {
    float block[16];
    for (uint k = 0; k < 16; ++k)
        block[k] = 0;
    for (uint ci = start; ci < metal::min(start + 256, cin); ++ci) {
        uint k = ci % 16;
        block[k] = metal::fma(X[row * cin + ci], W[co * cin + ci], block[k]);
    }
    for (uint k = 0; k < 16; ++k)
        parts[k] += block[k];
}
float acc = parts[0];
for (uint k = 1; k < 16; ++k)
    acc += parts[k];
Y[i] = acc + B[co];
"""

_OPS_PROJECTION = """

#pragma clang fp contract(off)
uint i = thread_position_in_grid.x, cin = W_shape[1], cout = W_shape[0];
if (i >= ROWS * cout)
    return;
uint row = i / cout, co = i % cout;
float first = 0, second = 0;
for (uint ci = 0; ci < cin / 2; ++ci)
    first = metal::fma(X[row * cin + ci], W[co * cin + ci], first);
for (uint ci = cin / 2; ci < cin; ++ci)
    second = metal::fma(X[row * cin + ci], W[co * cin + ci], second);
Y[i] = (first + B[co]) + second;
"""

_OPS_QKV = """

#pragma clang fp contract(off)
uint i = thread_position_in_grid.x, cin = W_shape[1], cout = W_shape[0];
if (i >= ROWS * cout)
    return;
uint row = i / cout, co = i % cout;
float sum = 0;
for (uint ci = 0; ci < cin; ++ci)
    sum = metal::fma(X[row * cin + ci], W[co * cin + ci], sum);
Y[i] = sum + B[co];
"""

_OPS_GEMV = """

#pragma clang fp contract(off)
uint co = thread_position_in_grid.x, cin = W_shape[1], cout = W_shape[0];
if (co >= cout)
    return;
float parts[P];
for (uint p = 0; p < P; ++p) {
    float s = 0;
    for (uint j = 0; j < SPAN; ++j) {
        uint ci = p * SPAN + j;
        if (ci < cin)
            s = metal::fma(X[ci], W[co * cin + ci], s);
    }
    parts[p] = s;
}
for (uint n = P / 2; n > 0; n /= 2)
    for (uint p = 0; p < n; ++p)
        parts[p] += parts[p + n];
Y[co] = parts[0] + B[co];
"""

_OPS_DOT = """

#pragma clang fp contract(off)
uint co = thread_position_in_grid.x;
if (co >= 128)
    return;
float blocks[2];
for (uint bl = 0; bl < 2; ++bl) {
    float p[128];
    for (uint j = 0; j < 128; ++j) {
        uint ci = bl * 256 + j * 2;
        float a = X[ci] * W[co * 512 + ci], b = X[ci + 1] * W[co * 512 + ci + 1];
        p[j] = a + b;
    }
    for (uint n = 64; n > 0; n /= 2)
        for (uint j = 0; j < n; ++j)
            p[j] += p[j + n];
    blocks[bl] = p[0];
}
Y[co] = (blocks[0] + blocks[1]) + B[co];
"""

_OPS_SLICED = """

#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
if (i >= ROWS * 512)
    return;
uint co = i % 512, row = i / 512;
float p[8];
for (uint j = 0; j < 8; ++j)
    p[j] = 0;
for (uint bl = 0; bl < 2; ++bl)
    for (uint a = 0; a < 4; ++a)
        for (uint ci = 0; ci < 64; ++ci) {
            uint kk = bl * 256 + (ci / 8) * 32 + a * 8 + ci % 8;
            p[bl * 4 + a] = metal::fma(X[row * 512 + kk], W[co * 512 + kk], p[bl * 4 + a]);
        }
float first = (p[0] + p[1]) + (p[2] + p[3]), second = (p[4] + p[5]) + (p[6] + p[7]);
Y[i] = (first + second) + B[co];
"""

_OPS_QKV4 = """

#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
if (i >= ROWS * 512)
    return;
uint co = i % 512, row = i / 512;
float p[4];
for (uint a = 0; a < 4; ++a) {
    float s = 0;
    for (uint ci = a * 128; ci < (a + 1) * 128; ++ci)
        s = metal::fma(X[row * 512 + ci], W[co * 512 + ci], s);
    p[a] = s;
}
Y[i] = ((p[0] + p[1]) + p[2]) + p[3] + B[co];
"""

_OPS_QKVS = """

#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
if (i >= ROWS * 512)
    return;
uint co = i % 512, row = i / 512;
float p[4];
for (uint a = 0; a < 4; ++a) {
    float s = 0;
    for (uint ci = 0; ci < 128; ++ci) {
        uint kk = (ci / 8) * 32 + a * 8 + ci % 8;
        s = metal::fma(X[row * 512 + kk], W[co * 512 + kk], s);
    }
    p[a] = s;
}
Y[i] = ((p[0] + p[1]) + (p[2] + p[3])) + B[co];
"""

_NORM_KERNEL = """

uint tid = thread_position_in_threadgroup.x;
uint lane = tid % 32;
uint warp = tid / 32;
uint row = threadgroup_position_in_grid.x;
threadgroup WF shared[32];
uint N = X_shape[1];
WF val = {0, 0, 0};
for (uint j = tid; j < N; j += TGS) {
    float data = X[row * N + j];
    float d = data - val.m;
    val.n += 1;
    val.m = val.m + d / val.n;
    val.v = metal::fma(d, data - val.m, val.v);
}
for (uint offset = 16; offset > 0; offset /= 2) {
    WF b = {simd_shuffle_down(val.m, offset), simd_shuffle_down(val.v, offset),
            simd_shuffle_down(val.n, offset)};
    val = combine(val, b);
}
if (lane == 0)
    shared[warp] = val;
threadgroup_barrier(mem_flags::mem_threadgroup);
val = tid < TGS / 32 ? shared[lane] : WF{0, 0, 0};
if (warp == 0) {
    for (uint offset = 16; offset > 0; offset /= 2) {
        WF b = {simd_shuffle_down(val.m, offset), simd_shuffle_down(val.v, offset),
                simd_shuffle_down(val.n, offset)};
        val = combine(val, b);
    }
}
if (tid == 0) {
    M[row] = val.m;
    uint bits = as_type<uint>(val.v / val.n + Eps[0]);
    int e = int(bits >> 23) - 127;
    uint index = (bits & 0x7fffff) | ((e & 1) << 23);
    int delta = -(e - (e & 1)) / 2;
    float value = val.v / val.n + Eps[0];
    R[row] =
        metal::isnan(value) || value < 0
            ? NAN
            : (value == 0 ? INFINITY
                          : (metal::isinf(value) ? 0
                                                 : as_type<float>(as_type<uint>(Table[index]) +
                                                                  uint(delta * 8388608))));
}
"""

_NORM_AFFINE = """

#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
uint hw = X_shape[2] * X_shape[3] * X_shape[4];
uint size = X_shape[0] * X_shape[1] * hw;
if (i >= size)
    return;
uint c = (i / hw) % X_shape[1];
uint ng = i / (hw * (X_shape[1] / G));
float a = R[ng] * W[c];
float b = B[c] - a * M[ng];
Y[i] = metal::fma(a, X[i], b);
"""

_MATH_KERNEL = """

uint i = thread_position_in_grid.x;
if (i >= X_SIZE)
    return;
float x = X[i];
float e = exp_cuda(SILU ? -x : x, Table);
Y[i] = SILU ? x / (1.f + e) : e;
"""

_MATH_SWISH = """

#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
if (i >= N)
    return;
float r = 1.f / (1.f + exp_cuda(-X[i], Table));
Y[i] = X[i] * r;
"""

_EMBED_KERNEL = """

#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
if (i >= N)
    return;
float x = X[i];
float q = metal::rint(x * as_type<float>(0x3f22f983u));
float r = metal::fma(q, as_type<float>(0xbfc90fdau), x);
r = metal::fma(q, as_type<float>(0xb3a22168u), r);
r = metal::fma(q, as_type<float>(0xa7c234c5u), r);
float z = r * r;
for (uint c = 0; c < 2; ++c) {
    int quadrant = int(q) + int(c);
    bool even = (quadrant & 1) == 0;
    float h = even ? r : 1.f;
    float a = even ? as_type<float>(0xb94d4153u)
                   : metal::fma(as_type<float>(0x37cbac00u), z, as_type<float>(0xbab607edu));
    float b = metal::fma(a, z, as_type<float>(even ? 0x3c0885e4u : 0x3d2aaabbu));
    float cc = metal::fma(b, z, as_type<float>(even ? 0xbe2aaaa8u : 0xbeffffffu));
    float y = metal::fma(cc, metal::fma(z, h, 0.f), h);
    if (quadrant & 2)
        y = metal::fma(y, -1.f, 0.f);
    if (c)
        C[i] = y;
    else
        S[i] = y;
}
"""

_ATTN_SOFTMAX = """

uint tid = thread_position_in_grid.x;
uint row = tid / 32, lane = tid % 32;
if (row >= ROWS)
    return;
float vals[ITS];
float maxval = -INFINITY;
for (uint i = 0; i < ITS; ++i) {
    uint k = lane + i * 32;
    float x = k < COLS ? X[row * COLS + k] : -INFINITY;
    vals[i] = x;
    maxval = metal::max(maxval, x);
}
for (uint d = 16; d > 0; d /= 2)
    maxval = metal::max(maxval, simd_shuffle_xor(maxval, d));
float sum = 0;
for (uint i = 0; i < ITS; ++i) {
    vals[i] = exp_cuda(vals[i] - maxval, Table);
    sum += vals[i];
}
for (uint d = 16; d > 0; d /= 2)
    sum += simd_shuffle_xor(sum, d);
for (uint i = 0; i < ITS; ++i) {
    uint k = lane + i * 32;
    if (k < COLS)
        Y[row * COLS + k] = vals[i] / sum;
}
"""

_ATTN_BLOCK = """

uint tid = thread_position_in_threadgroup.x, row = threadgroup_position_in_grid.x,
     lane = tid % 32, warp = tid / 32;
threadgroup float shared[32];
float vals[ITS], maxval = -INFINITY;
for (uint j = 0; j < ITS; ++j) {
    uint k = tid + j * 1024;
    vals[j] = k < COLS ? X[row * COLS + k] : -INFINITY;
    maxval = metal::max(maxval, vals[j]);
}
for (uint d = 16; d > 0; d /= 2)
    maxval = metal::max(maxval, simd_shuffle_down(maxval, d));
if (lane == 0)
    shared[warp] = maxval;
threadgroup_barrier(mem_flags::mem_threadgroup);
maxval = tid < 32 ? shared[lane] : -INFINITY;
if (warp == 0) {
    for (uint d = 16; d > 0; d /= 2)
        maxval = metal::max(maxval, simd_shuffle_down(maxval, d));
}
if (tid == 0)
    shared[0] = maxval;
threadgroup_barrier(mem_flags::mem_threadgroup);
maxval = shared[0];
float sum = 0;
for (uint j = 0; j < ITS; ++j) {
    vals[j] = exp_cuda(vals[j] - maxval, Table);
    sum += vals[j];
}
for (uint d = 16; d > 0; d /= 2)
    sum += simd_shuffle_down(sum, d);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0)
    shared[warp] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
sum = tid < 32 ? shared[lane] : 0;
if (warp == 0) {
    for (uint d = 16; d > 0; d /= 2)
        sum += simd_shuffle_down(sum, d);
}
if (tid == 0)
    shared[0] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
sum = shared[0];
for (uint j = 0; j < ITS; ++j) {
    uint k = tid + j * 1024;
    if (k < COLS)
        Y[row * COLS + k] = vals[j] / sum;
}
"""

KERNELS: dict[str, tuple[tuple[str, ...], tuple[str, ...], str, str]] = {
    "ops_conv": (("X", "W", "B"), ("Y",), "", _OPS_CONV),
    "ops_linear": (("X", "W", "B"), ("Y",), "", _OPS_LINEAR),
    "ops_projection": (("X", "W", "B"), ("Y",), "", _OPS_PROJECTION),
    "ops_qkv": (("X", "W", "B"), ("Y",), "", _OPS_QKV),
    "ops_gemv": (("X", "W", "B"), ("Y",), "", _OPS_GEMV),
    "ops_dot": (("X", "W", "B"), ("Y",), "", _OPS_DOT),
    "ops_sliced": (("X", "W", "B"), ("Y",), "", _OPS_SLICED),
    "ops_qkv4": (("X", "W", "B"), ("Y",), "", _OPS_QKV4),
    "ops_qkvs": (("X", "W", "B"), ("Y",), "", _OPS_QKVS),
    "norm_kernel": (("X", "Eps", "Table"), ("M", "R"), _HEADER_0, _NORM_KERNEL),
    "norm_affine": (("X", "M", "R", "W", "B"), ("Y",), "", _NORM_AFFINE),
    "math_kernel": (("X", "Table"), ("Y",), _HEADER_1, _MATH_KERNEL),
    "math_swish": (("X", "Table"), ("Y",), _HEADER_1, _MATH_SWISH),
    "embed_kernel": (("X",), ("C", "S"), "", _EMBED_KERNEL),
    "attn_softmax": (("X", "Table"), ("Y",), _HEADER_1, _ATTN_SOFTMAX),
    "attn_block": (("X", "Table"), ("Y",), _HEADER_1, _ATTN_BLOCK),
}

_VAE_REDUCE_HEADER = """
#pragma clang fp contract(off)
struct VAEState {
  float m, v, n;
};
VAEState vae_combine(VAEState a, VAEState b, bool variance) {
  if (!variance)
    return {a.m + b.m, 0, 0};
  if (a.n == 0)
    return b;
  if (b.n == 0)
    return a;
  float d = b.m - a.m;
  float n = a.n + b.n;
  float p = b.n / n;
  return {metal::fma(d, p, a.m), metal::fma(d * d * a.n, p, a.v + b.v), n};
}
VAEState vae_add(VAEState a, float x, bool variance) {
  if (!variance)
    return {a.m + x, 0, 0};
  float n = a.n + 1;
  float delta = x - a.m;
  float m = a.m + delta / n;
  return {m, metal::fma(delta, x - m, a.v), n};
}
"""

_VAE_REDUCE_PARTIAL = """
uint tid = thread_position_in_threadgroup.x;
uint block = threadgroup_position_in_grid.x, row = block / CTAS, cta = block % CTAS,
     n = X_shape[1];
VAEState parts[VEC];
for (uint k = 0; k < VEC; ++k)
  parts[k] = {0, 0, 0};
uint stride = TGS * CTAS;
if (VECTOR) {
  for (uint i = tid + cta * TGS; i * VEC + VEC - 1 < n; i += stride) {
    for (uint k = 0; k < VEC; ++k)
      parts[k] = vae_add(parts[k], X[row * n + i * VEC + k], VARIANCE);
  }
  uint tail = n - n % VEC + tid;
  if (cta == 0 && tail < n)
    parts[0] = vae_add(parts[0], X[row * n + tail], VARIANCE);
} else {
  for (uint k = 0; k < VEC; ++k) {
    for (uint i = tid + cta * TGS + k * stride; i < n; i += VEC * stride)
      parts[k] = vae_add(parts[k], X[row * n + i], VARIANCE);
  }
}
VAEState value = parts[0];
for (uint k = 1; k < VEC; ++k)
  value = vae_combine(value, parts[k], VARIANCE);
threadgroup VAEState shared[TGS];
shared[tid] = value;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint offset = TGS / 2; offset > 0; offset /= 2) {
  if (tid < offset)
    shared[tid] = vae_combine(shared[tid], shared[tid + offset], VARIANCE);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (tid == 0) {
  Y[block * 3] = shared[0].m;
  Y[block * 3 + 1] = shared[0].v;
  Y[block * 3 + 2] = shared[0].n;
}
"""

_VAE_REDUCE_FINAL = """
uint tid = thread_position_in_threadgroup.x, row = threadgroup_position_in_grid.x;
VAEState value = {0, 0, 0};
for (uint i = tid; i < CTAS; i += TGS) {
  uint at = (row * CTAS + i) * 3;
  value = vae_combine(value, {X[at], X[at + 1], X[at + 2]}, VARIANCE);
}
threadgroup VAEState shared[TGS];
shared[tid] = value;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint offset = TGS / 2; offset > 0; offset /= 2) {
  if (tid < offset)
    shared[tid] = vae_combine(shared[tid], shared[tid + offset], VARIANCE);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (tid == 0)
  Y[row] = VARIANCE ? shared[0].v / shared[0].n : shared[0].m * (1.f / float(N));
"""

_VAE_TRILINEAR = """
#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
uint w = X_shape[4], h = X_shape[3], d = X_shape[2];
uint ow = w * 2, oh = h * 2, od = d * 2;
uint vol = w * h * d, ovol = vol * 8;
if (i >= X_shape[0] * X_shape[1] * ovol)
  return;
uint ox = i % ow, oy = (i / ow) % oh, oz = (i / (ow * oh)) % od, nc = i / ovol;
float fx = metal::max(0.f, float(ox) * .5f - .25f),
      fy = metal::max(0.f, float(oy) * .5f - .25f),
      fz = metal::max(0.f, float(oz) * .5f - .25f);
uint x0 = uint(fx), y0 = uint(fy), z0 = uint(fz), x1 = metal::min(x0 + 1, w - 1),
     y1 = metal::min(y0 + 1, h - 1), z1 = metal::min(z0 + 1, d - 1);
float ax = fx - x0, ay = fy - y0, az = fz - z0;
float bx = 1 - ax, by = 1 - ay, bz = 1 - az;
float row[4];
uint ys[2] = {y0, y1}, zs[2] = {z0, z1};
for (uint z = 0; z < 2; ++z) {
  for (uint y = 0; y < 2; ++y) {
    uint base = nc * vol + zs[z] * h * w + ys[y] * w;
    row[z * 2 + y] = metal::fma(bx, X[base + x0], ax * X[base + x1]);
  }
}
float plane0, plane1;
plane0 = metal::fma(by, row[0], ay *row[1]);
plane1 = metal::fma(by, row[2], ay *row[3]);
Y[i] = metal::fma(bz, plane0, az *plane1);
"""

KERNELS.update(
    {
        "vae_reduce_partial": (("X",), ("Y",), _VAE_REDUCE_HEADER, _VAE_REDUCE_PARTIAL),
        "vae_reduce_final": (("X",), ("Y",), _VAE_REDUCE_HEADER, _VAE_REDUCE_FINAL),
        "vae_trilinear": (("X",), ("Y",), "", _VAE_TRILINEAR),
    }
)
