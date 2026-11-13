import pathlib

p = pathlib.Path("scripts/xval_projection_laplace/oss_ports.py")
s = p.read_text()

subs = [
    ("    pos = rng.uniform(-3.0, 3.0, size=(n, 2)) * h\n",
     "    pos = np.zeros((n, 3))\n    pos[:, :2] = rng.uniform(-3.0, 3.0, size=(n, 2)) * h\n"),
    ("    grad = torch.zeros((n, 2), dtype=pos.dtype)\n",
     "    grad = torch.zeros((n, pos.shape[1]), dtype=pos.dtype)\n"),
    ("    dcx, dcy = zeros.copy(), zeros.copy()\n",
     "    dcx, dcy, dcz = zeros.copy(), zeros.copy(), zeros.copy()\n"),
    ("                 s_scolor=d_scolor, d_cx=dcx, d_cy=dcy, d_cz=zeros,\n",
     "                 s_scolor=d_scolor, d_cx=dcx, d_cy=dcy, d_cz=dcz,\n"),
    ("    dcx2, ddd2 = zeros.copy(), zeros.copy()\n",
     "    dcx2, dcy2, dcz2, ddd2 = (zeros.copy(), zeros.copy(), zeros.copy(),\n"
     "                              zeros.copy())\n"),
    ("                 d_cx=dcx2, d_cy=zeros, d_cz=zeros, d_V=vol, s_V=vol,\n",
     "                 d_cx=dcx2, d_cy=dcy2, d_cz=dcz2, d_V=vol, s_V=vol,\n"),
    ("        eqd.post_loop(d_idx=a, d_cx=dcx2, d_cy=zeros, d_cz=zeros, d_nx=dnx,\n",
     "        eqd.post_loop(d_idx=a, d_cx=dcx2, d_cy=dcy2, d_cz=dcz2, d_nx=dnx,\n"),
    ("        np.max(np.abs(div_lin - 2.0)))",
     "        np.max(np.abs(div_lin - float(pos.shape[1]))))"),
]
for a, b in subs:
    if a not in s:
        raise SystemExit("MISSING: " + a[:60])
    s = s.replace(a, b, 1)
p.write_text(s)
print("patched ok")