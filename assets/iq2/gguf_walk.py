"""Hand-rolled GGUF reader -- no gguf library, just struct.  Mirrors the layout in ggml/include/gguf.h.

    python gguf_walk.py model.gguf
"""
import collections, math, os, struct, sys

# ggml_type id -> (name, block_size_in_weights, bytes_per_block)   (ggml/src/ggml.c type_traits)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2),
}
# gguf_type id -> struct format for scalars (8=STRING, 9=ARRAY handled separately)
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Reader:
    def __init__(self, f): self.f = f
    def unpack(self, fmt):
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]
    def string(self):
        n = self.unpack("<Q")                      # uint64 length, no NUL terminator
        return self.f.read(n).decode("utf-8", errors="replace")
    def value(self, vtype):
        if vtype == 8:
            return self.string()
        if vtype == 9:                             # ARRAY: elem type, count, elems
            etype, n = self.unpack("<I"), self.unpack("<Q")
            return [self.value(etype) for _ in range(n)]
        return self.unpack(SCALAR[vtype])


def read_gguf(path):
    with open(path, "rb") as f:
        r = Reader(f)
        magic = f.read(4)
        assert magic == b"GGUF", magic
        version = r.unpack("<I")
        n_tensors, n_kv = r.unpack("<q"), r.unpack("<q")

        kv = collections.OrderedDict()
        for _ in range(n_kv):
            key = r.string()
            vtype = r.unpack("<I")
            kv[key] = (vtype, r.value(vtype))

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.unpack("<I")
            ne = [r.unpack("<q") for _ in range(n_dims)]   # ne[0] is the contiguous (fastest) dim
            ttype = r.unpack("<I")
            offset = r.unpack("<Q")                        # relative to start of data blob
            tname, blck, tsize = GGML_TYPES[ttype]
            nel = 1
            for d in ne: nel *= d
            nbytes = nel // blck * tsize
            tensors.append(dict(name=name, ne=ne, type=tname, type_id=ttype, offset=offset, nbytes=nbytes))

        end_of_header = f.tell()
        align = kv.get("general.alignment", (4, 32))[1]
        data_offset = (end_of_header + align - 1) // align * align
    return dict(version=version, kv=kv, tensors=tensors, header_bytes=end_of_header,
                align=align, data_offset=data_offset, file_size=os.path.getsize(path))


def show(path):
    g = read_gguf(path)
    print(f"== {os.path.basename(path)}")
    print(f"magic=GGUF version={g['version']} n_tensors={len(g['tensors'])} n_kv={len(g['kv'])}")
    print(f"header+kv+tensor-info = {g['header_bytes']:,} B; alignment={g['align']} -> data blob starts at {g['data_offset']:,}")
    print("\n-- metadata (first 3 elems of arrays) --")
    for k, (vt, v) in g["kv"].items():
        if isinstance(v, list):
            v = f"[{', '.join(map(repr, v[:3]))}, ...] (len {len(v)})"
        elif isinstance(v, str) and len(v) > 60:
            v = repr(v[:40]) + f"... ({len(v)} chars)"
        print(f"  {k:45s} {v}")
    print("\n-- tensor infos (first 14) --")
    print(f"  {'name':28s} {'ne (ne0 first)':>20s} {'type':>8s} {'offset':>14s} {'bytes':>12s}")
    for t in g["tensors"][:14]:
        print(f"  {t['name']:28s} {str(t['ne']):>20s} {t['type']:>8s} {t['offset']:>14,} {t['nbytes']:>12,}")
    hist = collections.Counter(t["type"] for t in g["tensors"])
    by_bytes = collections.Counter()
    for t in g["tensors"]: by_bytes[t["type"]] += t["nbytes"]
    print("\n-- tensor type mix --")
    for ty, n in hist.most_common():
        print(f"  {ty:8s} {n:4d} tensors  {by_bytes[ty]/2**20:9.1f} MiB")
    last = max(g["tensors"], key=lambda t: t["offset"])
    print(f"\ncheck: data_offset + last.offset + last.nbytes = {g['data_offset'] + last['offset'] + last['nbytes']:,}  "
          f"file size = {g['file_size']:,}")
    n_w = sum(math.prod(t["ne"]) for t in g["tensors"])
    print(f"bits per weight over all tensors = {8*sum(t['nbytes'] for t in g['tensors'])/n_w:.4f}")
    return g


if __name__ == "__main__":
    show(sys.argv[1])
