"""Readable Python port of llama.cpp's IQ2_XXS quantizer (ggml-quants.c: iq2xs_init_impl,
quantize_row_iq2_xxs_impl, make_qp_quants, iq2_find_best_neighbour).

    python iq2xxs_quant.py quant.gguf orig_bf16.gguf imatrix.gguf tensor_name n_rows
compares our bytes with llama-quantize's bytes for the first n_rows rows of a tensor.
"""
import math, struct, sys
import numpy as np
from tables import iq2xxs_grid, grid_bytes

# ---------------------------------------------------------------- tables (iq2xs_init_impl)
# Encoder works in "level space": each coordinate is l in {0,1,2}, value q = 2l+1 in {1,3,5}.
LEVELS = {8: 0, 25: 1, 43: 2}
GRID_L = [[LEVELS[b] for b in grid_bytes(g)] for g in iq2xxs_grid]          # 256 x 8 levels
code = lambda ls: sum(l << 2 * j for j, l in enumerate(ls))                 # pack 8 levels, 2 bits each
KMAP = {code(ls): i for i, ls in enumerate(GRID_L)}                         # 8-level code -> grid index
_NEIGH = {}

def neighbours(u):
    """Grid points in the 2 nearest distance shells of an off-grid code u (nwant = 2)."""
    if u not in _NEIGH:
        pos = [2 * ((u >> 2 * j) & 3) + 1 for j in range(8)]
        d2 = sorted((sum((2 * a + 1 - p) ** 2 for a, p in zip(g, pos)), i) for i, g in enumerate(GRID_L))
        shells = sorted({d for d, _ in d2})[:2]
        _NEIGH[u] = [i for d, i in d2 if d in shells]
    return _NEIGH[u]

def nearest_int(f):
    return int(round(f))                                                    # round-half-even, like the C trick

def best_neighbour(u, xval, waux, scale):
    """iq2_find_best_neighbour: weighted (by sqrt(weight)) closest neighbour at this scale."""
    best, best_i = math.inf, -1
    for i in neighbours(u):
        d2 = sum(w * (scale * (2 * l + 1) - x) ** 2 for l, x, w in zip(GRID_L[i], xval, waux))
        if d2 < best:
            best, best_i = d2, i
    return best_i

def snap(xval8, waux8, id_, scale):
    """Round 8 magnitudes to levels at inverse-scale id_, then force the pattern onto the grid."""
    ls = [max(0, min(2, nearest_int(0.5 * (id_ * x - 1)))) for x in xval8]
    gi = KMAP.get(code(ls))
    if gi is None:
        gi = best_neighbour(code(ls), xval8, waux8, scale)
    return gi, GRID_L[gi]

def make_qp_quants(nmax, x, w):
    """Initial scale guess: weighted LSQ fit of x ~ scale * L with integer L in [0, nmax].
    Quirk kept for bit-exactness: C stores L in uint8_t, and x may hold one negative value (the
    parity-flipped weight), so a negative level wraps to 256+l wherever L is read back."""
    u8 = lambda l: l & 0xFF
    mx = max(x)
    iscale = nmax / mx
    L = [u8(nearest_int(iscale * v)) for v in x]
    best = sum(wi * (v - l / iscale) ** 2 for v, l, wi in zip(x, L, w))
    for i_s in [-4, -3, -2, -1, 1, 2, 3, 4]:
        isc = (0.1 * i_s + nmax) / mx
        mse = sum(wi * (v - min(nmax, nearest_int(isc * v)) / isc) ** 2 for v, wi in zip(x, w))
        if mse < best:
            best, iscale = mse, isc
    Lint = [min(nmax, nearest_int(iscale * v)) for v in x]                 # sums use the signed level...
    slx = sum(wi * v * l for v, l, wi in zip(x, Lint, w)); sl2 = sum(wi * l * l for l, wi in zip(Lint, w))
    L = [u8(l) for l in Lint]                                              # ...but L[] keeps the wrapped one
    for _ in range(5):                                                     # coordinate-descent refinement
        changed = 0
        for i in range(len(x)):
            a = slx - w[i] * x[i] * L[i]; b = sl2 - w[i] * L[i] * L[i]
            if a > 0 and b > 0:
                nl = min(nmax, nearest_int(x[i] * b / a))
                if nl != L[i]:
                    a += w[i] * x[i] * nl; b += w[i] * nl * nl
                    if a * a * sl2 > slx * slx * b:
                        L[i], slx, sl2, changed = u8(nl), a, b, changed + 1
        if not changed:
            break
    return slx / sl2 if sl2 > 0 else 0.0

# ---------------------------------------------------------------- one 256-weight block
def quantize_block(x, qw):
    sigma2 = sum(v * v for v in x) / 256
    scales, q2 = [0.0] * 8, [0] * 16
    for ib in range(8):
        xb, qb = x[32 * ib:32 * ib + 32], qw[32 * ib:32 * ib + 32]
        weight = [q * math.sqrt(sigma2 + v * v) for q, v in zip(qb, xb)]   # importance x magnitude
        waux = [math.sqrt(w) for w in weight]

        # 1. signs: store |x|; if a group of 8 has an odd number of negatives, flip the cheapest one
        xval, signs = [abs(v) for v in xb], []
        for k in range(4):
            s = sum(1 << i for i in range(8) if xb[8 * k + i] < 0)
            if bin(s).count("1") % 2:
                imin = min(range(8), key=lambda i: (weight[8 * k + i] * xb[8 * k + i] ** 2, i))
                xval[8 * k + imin] = -xval[8 * k + imin]
                s ^= 1 << imin
            signs.append(s & 127)                                          # 8th bit is implied by parity
        if max(xval) < 1e-15:
            continue

        # 2. scale search: 13 candidate inverse scales around the make_qp_quants guess
        scale = make_qp_quants(4, xval, weight)
        eff_max = scale * 3                                                # top level (q=5) ~ max|x|
        if eff_max <= 0:
            continue
        best = 0.0
        for i_s in range(-6, 7):
            id_ = (5 + 0.1 * i_s) / eff_max
            L = sum((snap(xval[8 * k:8 * k + 8], waux[8 * k:8 * k + 8], id_, 1 / id_)[1] for k in range(4)), [])
            sqx = sum(w * v * (2 * l + 1) for w, v, l in zip(weight, xval, L))
            sq2 = sum(w * (2 * l + 1) ** 2 for w, l in zip(weight, L))
            if sq2 > 0 and sqx * sqx > best * sq2:                         # maximizes sqx^2/sq2 == min weighted MSE
                scale, best = sqx / sq2, sqx * sqx / sq2

        # 3. final snap at the best scale, then refit the scale for the chosen grid points
        gidx, L = [], []
        for k in range(4):
            gi, ls = snap(xval[8 * k:8 * k + 8], waux[8 * k:8 * k + 8], 1 / scale, scale)
            gidx.append(gi); L += ls
        scale = sum(w * v * (2 * l + 1) for w, v, l in zip(weight, xval, L)) / \
                sum(w * (2 * l + 1) ** 2 for w, l in zip(weight, L))

        for k in range(4):
            q2[2 * ib] |= gidx[k] << 8 * k
            q2[2 * ib + 1] |= signs[k] << 7 * k
        scales[ib] = scale

    max_scale = max(scales)
    if max_scale == 0:
        return bytes(66)
    d = max_scale / 31                                                     # super-block scale
    for ib in range(8):                                                    # 4-bit sub-scale: scale ~ d*(2l+1)
        q2[2 * ib + 1] |= max(0, min(15, nearest_int(0.5 * (scales[ib] / d - 1)))) << 28
    return np.float16(d).tobytes() + struct.pack("<16I", *q2)


if __name__ == "__main__":
    from gguf_walk import read_gguf
    from iq2_decode import raw, bf16_weights
    qpath, opath, ipath, name, nrows = sys.argv[1:6]
    nrows = int(nrows)
    im = read_gguf(ipath)
    get = lambda n: raw(ipath, im, next(t for t in im["tensors"] if t["name"] == n)).view(np.float32)
    qw = (get(name + ".in_sum2") / get(name + ".counts")[0]).astype(np.float64).tolist()
    g = read_gguf(qpath); t = next(t for t in g["tensors"] if t["name"] == name)
    ne0 = t["ne"][0]
    ref = raw(qpath, g, t).reshape(-1, 66)
    W = bf16_weights(opath, name).reshape(-1, ne0)
    same = scale_only = total = 0
    for r in range(nrows):
        row = W[r].astype(np.float64).tolist()
        for b in range(ne0 // 256):
            mine = np.frombuffer(quantize_block(row[256 * b:256 * b + 256], qw[256 * b:256 * b + 256]), np.uint8)
            theirs = ref[r * (ne0 // 256) + b]
            total += 1
            if np.array_equal(mine, theirs):
                same += 1
            elif np.array_equal(mine[2:], theirs[2:]):
                scale_only += 1
    print(f"{name}: {same}/{total} blocks byte-identical to llama-quantize "
          f"({100 * same / total:.2f}%), {scale_only} differ only in the fp16 d")
