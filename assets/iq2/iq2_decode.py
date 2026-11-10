"""Decode IQ2_XXS / IQ2_XS / IQ2_S blocks by hand (ported from ggml-quants.c dequantize_row_iq2_*),
check against gguf-py's reference dequantizer, and compare with the original BF16 weights.

    python iq2_decode.py quant.gguf orig_bf16.gguf [tensor_name]
"""
import sys, struct
import numpy as np
from gguf_walk import read_gguf
from tables import iq2xxs_grid, iq2xs_grid, iq2s_grid, ksigns, grid_bytes

GRID = {t: np.array([grid_bytes(g) for g in tab], dtype=np.float32)
        for t, tab in [("IQ2_XXS", iq2xxs_grid), ("IQ2_XS", iq2xs_grid), ("IQ2_S", iq2s_grid)]}
KSIGNS = np.array(ksigns, dtype=np.uint8)
BITS = np.arange(8, dtype=np.uint8)                      # kmask_iq2xs[j] = 1 << j


def sign_bytes_to_pm1(sb):                               # (..., ) uint8 -> (..., 8) of +-1
    return np.where((sb[..., None] >> BITS) & 1, -1.0, 1.0).astype(np.float32)


def deq_iq2_xxs(blk):                                    # blk: (nb, 66) uint8
    nb = blk.shape[0]
    d = blk[:, :2].copy().view(np.float16).astype(np.float32)          # (nb,1)
    q = blk[:, 2:].copy().view(np.uint32).reshape(nb, 8, 2)            # 8 sub-blocks x (aux32[0], aux32[1])
    db = d * (0.5 + (q[..., 1] >> 28)) * 0.25                          # (nb,8)  4-bit scale in top nibble
    idx = q[..., 0:1].view(np.uint8).reshape(nb, 8, 4)                 # 4 x 8-bit grid indices
    s7 = (q[..., 1:2] >> np.array([0, 7, 14, 21], dtype=np.uint32)) & 127   # 4 x 7-bit sign codes
    signs = sign_bytes_to_pm1(KSIGNS[s7])                              # parity restores the 8th sign
    return (db[..., None, None] * GRID["IQ2_XXS"][idx] * signs).reshape(nb, 256)


def deq_iq2_xs(blk):                                     # (nb, 74): d | qs[32] u16 | scales[8]
    nb = blk.shape[0]
    d = blk[:, :2].copy().view(np.float16).astype(np.float32)
    qs = blk[:, 2:66].copy().view(np.uint16).reshape(nb, 8, 4)
    sc = blk[:, 66:74]
    db = d[..., None] * (0.5 + np.stack([sc & 15, sc >> 4], -1)) * 0.25   # (nb,8,2): one scale per 16
    db = np.repeat(db, 2, axis=-1)                                     # groups l=0,1 -> lo, l=2,3 -> hi
    signs = sign_bytes_to_pm1(KSIGNS[qs >> 9])                         # 9-bit index | 7-bit signs
    return (db[..., None] * GRID["IQ2_XS"][qs & 511] * signs).reshape(nb, 256)


def deq_iq2_s(blk):                                      # (nb, 82): d | qs[32] | signs[32] | qh[8] | scales[8]
    nb = blk.shape[0]
    d = blk[:, :2].copy().view(np.float16).astype(np.float32)
    lo = blk[:, 2:34].reshape(nb, 8, 4).astype(np.uint16)
    sg = blk[:, 34:66].reshape(nb, 8, 4)                               # full 8 sign bits, no parity trick
    qh = blk[:, 66:74].astype(np.uint16)
    sc = blk[:, 74:82]
    hi = (qh[..., None] << (8 - 2 * np.arange(4, dtype=np.uint16))) & 0x300
    db = np.repeat(d[..., None] * (0.5 + np.stack([sc & 15, sc >> 4], -1)) * 0.25, 2, axis=-1)
    return (db[..., None] * GRID["IQ2_S"][lo | hi] * sign_bytes_to_pm1(sg)).reshape(nb, 256)


DEQ = {"IQ2_XXS": (deq_iq2_xxs, 66), "IQ2_XS": (deq_iq2_xs, 74), "IQ2_S": (deq_iq2_s, 82)}


def raw(path, g, t):
    with open(path, "rb") as f:
        f.seek(g["data_offset"] + t["offset"])
        return np.frombuffer(f.read(t["nbytes"]), dtype=np.uint8)


def bf16_weights(path, name):
    g = read_gguf(path); t = next(t for t in g["tensors"] if t["name"] == name)
    assert t["type"] == "BF16"
    u16 = raw(path, g, t).view(np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


def walkthrough_xxs(blk, orig):
    """Verbose decode of one IQ2_XXS block, sub-block 0, exactly like the C loop."""
    print(f"raw block (66 B): {blk[:2].tobytes().hex()} | {blk[2:10].tobytes().hex()} | {blk[10:].tobytes().hex()[:24]}...")
    d = float(blk[:2].copy().view(np.float16)[0])
    print(f"  d (fp16 super-block scale) = {d:.6g}")
    aux0, aux1 = struct.unpack("<II", blk[2:10].tobytes())
    print(f"  sub-block 0: aux32[0]=0x{aux0:08x}  aux32[1]=0x{aux1:08x}")
    print(f"     aux32[1] bits = {aux1 >> 28:04b} | {(aux1 >> 21) & 127:07b} {(aux1 >> 14) & 127:07b} "
          f"{(aux1 >> 7) & 127:07b} {aux1 & 127:07b}   (scale | signs g3 g2 g1 g0)")
    l = aux1 >> 28
    db = d * (0.5 + l) * 0.25
    print(f"     4-bit scale l={l} -> db = d*(0.5+l)/4 = {db:.6g}")
    for k in range(4):
        gi = (aux0 >> (8 * k)) & 0xFF
        s7 = (aux1 >> (7 * k)) & 127
        s8 = ksigns[s7]
        mags = grid_bytes(iq2xxs_grid[gi])
        vals = [db * m * (-1 if (s8 >> j) & 1 else 1) for j, m in enumerate(mags)]
        print(f"     group {k}: grid idx {gi:3d} -> {mags}  signs7={s7:07b} -> ksigns={s8:08b}")
        print(f"              decoded : {' '.join(f'{v:+.4f}' for v in vals)}")
        print(f"              original: {' '.join(f'{v:+.4f}' for v in orig[8*k:8*k+8])}")


def main(qpath, opath, focus=None):
    g = read_gguf(qpath)
    print(f"{'tensor':28s} {'type':8s} {'rel-RMSE':>9s} {'cos':>8s}  max|mine - gguf-py|")
    import gguf.quants as GQ
    from gguf import GGMLQuantizationType as QT
    for t in g["tensors"]:
        if t["type"] not in DEQ:
            continue
        fn, bs = DEQ[t["type"]]
        blocks = raw(qpath, g, t).reshape(-1, bs)
        mine = fn(blocks).reshape(-1)
        ref = GQ.dequantize(raw(qpath, g, t), QT[t["type"]]).reshape(-1)
        orig = bf16_weights(opath, t["name"])
        err = np.sqrt(np.mean((mine - orig) ** 2) / np.mean(orig ** 2))
        cos = mine @ orig / np.linalg.norm(mine) / np.linalg.norm(orig)
        if t["name"].startswith(("blk.0.", "blk.13.", "blk.27.")) or t["name"] == focus:
            print(f"{t['name']:28s} {t['type']:8s} {err:9.4f} {cos:8.4f}  {np.abs(mine - ref).max():.3g}")
        assert np.array_equal(mine, ref), t["name"]
    print("all IQ2 tensors: hand decoder == gguf-py reference, bit-exact")

    if focus:
        t = next(t for t in g["tensors"] if t["name"] == focus)
        assert t["type"] == "IQ2_XXS", t["type"]
        print(f"\n=== walkthrough: {focus}  ne={t['ne']}  -> {t['ne'][0]//256} blocks per row, "
              f"{t['ne'][0]//256*66} B per row")
        blocks = raw(qpath, g, t).reshape(-1, 66)
        walkthrough_xxs(blocks[0], bf16_weights(opath, focus)[:32])

        # Experiment: the encoder searches with levels {1,3,5}; the decode grid uses {8,25,43}/8.
        orig = bf16_weights(opath, focus)
        for name, lv in [("shipped 8/25/43", None), ("'consistent' 8/24/40", {8: 8, 25: 24, 43: 40})]:
            saved = GRID["IQ2_XXS"].copy()
            if lv: GRID["IQ2_XXS"] = np.vectorize(lv.get)(saved).astype(np.float32)
            w = deq_iq2_xxs(blocks).reshape(-1)
            GRID["IQ2_XXS"] = saved
            print(f"  decode grid {name:22s}: rel-RMSE = {np.sqrt(np.mean((w-orig)**2)/np.mean(orig**2)):.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
