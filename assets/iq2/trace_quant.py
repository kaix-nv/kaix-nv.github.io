"""Trace the IQ2_XXS quantizer through one real sub-block, then gather statistics over many blocks.

    python trace_quant.py orig_bf16.gguf imatrix.gguf [tensor_name]
"""
import itertools, math, sys, collections
import numpy as np
from iq2xxs_quant import KMAP, code, neighbours, nearest_int, make_qp_quants, snap, quantize_block
from gguf_walk import read_gguf
from iq2_decode import raw, bf16_weights

BF, IMAT = sys.argv[1], sys.argv[2]
NAME = sys.argv[3] if len(sys.argv) > 3 else "blk.13.ffn_up.weight"
im = read_gguf(IMAT)
get = lambda n: raw(IMAT, im, next(t for t in im["tensors"] if t["name"] == n)).view(np.float32)
qw_all = (get(NAME + ".in_sum2") / get(NAME + ".counts")[0]).astype(np.float64)
W = bf16_weights(BF, NAME).reshape(-1, qw_all.size).astype(np.float64)

# ---------------- table statistics
offgrid = [code(ls) for ls in itertools.product(range(3), repeat=8) if code(ls) not in KMAP]
nl = [len(neighbours(u)) for u in offgrid]
print(f"level patterns: 3^8 = {3**8}; on grid: {len(KMAP)}; off grid: {len(offgrid)}; "
      f"neighbour list length min/mean/max = {min(nl)}/{np.mean(nl):.1f}/{max(nl)}")

# ---------------- trace sub-block 0 of row 0, block 0
x = W[0, :256].tolist(); qw = qw_all[:256].tolist()
sigma2 = sum(v * v for v in x) / 256
xb, qb = x[:32], qw[:32]
weight = [q * math.sqrt(sigma2 + v * v) for q, v in zip(qb, xb)]
waux = [math.sqrt(w) for w in weight]
print(f"\nsigma2 = {sigma2:.4g} (rms {math.sqrt(sigma2):.4g});  imatrix E[a^2] for cols 0..7: "
      f"{' '.join(f'{q:.3g}' for q in qb[:8])}")
xval = [abs(v) for v in xb]
for k in range(4):
    neg = [i for i in range(8) if xb[8 * k + i] < 0]
    s = sum(1 << i for i in neg)
    msg = ""
    if len(neg) % 2:
        imin = min(range(8), key=lambda i: (weight[8 * k + i] * xb[8 * k + i] ** 2, i))
        xval[8 * k + imin] = -xval[8 * k + imin]; s ^= 1 << imin
        msg = f"odd -> flip w[{imin}] (x={xb[8*k+imin]:+.6f}, cost w*x^2={weight[8*k+imin]*xb[8*k+imin]**2:.2e})"
    print(f"group {k}: negatives at {neg} {msg}  -> stored 7 bits {s & 127:07b}")
s0 = make_qp_quants(4, xval, weight)
eff_max = 3 * s0
print(f"\nmax|x| = {max(xval):.5f}; make_qp_quants scale = {s0:.6f} -> eff_max = {eff_max:.6f}")
print(f"{'is':>3s} {'id':>8s} {'1/id':>9s} {'off-grid':>8s} {'LSQ scale':>10s} {'score':>10s}")
best = 0
for i_s in range(-6, 7):
    id_ = (5 + 0.1 * i_s) / eff_max
    L, off = [], 0
    for k in range(4):
        ls = [max(0, min(2, nearest_int(0.5 * (id_ * v - 1)))) for v in xval[8 * k:8 * k + 8]]
        off += code(ls) not in KMAP
        L += snap(xval[8 * k:8 * k + 8], waux[8 * k:8 * k + 8], id_, 1 / id_)[1]
    sqx = sum(w * v * (2 * l + 1) for w, v, l in zip(weight, xval, L))
    sq2 = sum(w * (2 * l + 1) ** 2 for w, l in zip(weight, L))
    score = sqx * sqx / sq2
    mark = "  <- best so far" if score > best else ""
    best = max(best, score)
    print(f"{i_s:3d} {id_:8.2f} {1/id_:9.6f} {off:8d} {sqx/sq2:10.6f} {score:10.4g}{mark}")

# ---------------- statistics over the first 64 rows
groups = odd = 0
l_hist = collections.Counter()
nb = W.shape[1] // 256
for r in range(64):
    for b in range(nb):
        x = W[r, 256 * b:256 * b + 256].tolist()
        out = np.frombuffer(quantize_block(x, qw_all[256 * b:256 * b + 256].tolist()), np.uint8)
        q = out[2:].view(np.uint32).reshape(8, 2)
        for ib in range(8):
            l_hist[int(q[ib, 1] >> 28)] += 1
            for k in range(4):
                g = x[32 * ib + 8 * k:32 * ib + 8 * k + 8]
                groups += 1
                odd += sum(v < 0 for v in g) % 2
print(f"\n{groups} groups: {100 * odd / groups:.1f}% needed a parity sign flip")
print("4-bit sub-block scale l histogram:", dict(sorted(l_hist.items())))
