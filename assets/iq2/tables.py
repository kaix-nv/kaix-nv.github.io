"""Parse the IQ2 lookup tables straight out of llama.cpp's ggml-common.h.
Set LLAMA_CPP to your llama.cpp checkout (default: ./llama.cpp)."""
import os, re
HDR = os.path.join(os.environ.get("LLAMA_CPP", "llama.cpp"), "ggml", "src", "ggml-common.h")
_src = open(HDR).read()

def table(name):
    m = re.search(r"GGML_TABLE_BEGIN\(\w+,\s*%s,\s*(\d+)\)(.*?)GGML_TABLE_END" % name, _src, re.S)
    n, body = int(m.group(1)), m.group(2)
    vals = [int(t, 0) for t in re.findall(r"0x[0-9a-fA-F]+|\d+", body)]
    assert len(vals) == n, (name, len(vals), n)
    return vals

iq2xxs_grid = table("iq2xxs_grid")   # 256 x uint64
iq2xs_grid  = table("iq2xs_grid")    # 512 x uint64
iq2s_grid   = table("iq2s_grid")     # 1024 x uint64
ksigns      = table("ksigns_iq2xs")  # 128 x uint8
kmask       = [1 << j for j in range(8)]  # kmask_iq2xs

def grid_bytes(g):
    """uint64 grid entry -> 8 magnitudes, little-endian byte order (byte 0 = weight 0)."""
    return [(g >> (8 * j)) & 0xFF for j in range(8)]
