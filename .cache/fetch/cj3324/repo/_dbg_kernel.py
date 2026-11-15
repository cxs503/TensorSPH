import sys
sys.path.insert(0, "/tmp/xv2/lib64/python3.11/site-packages")
sys.path.insert(0, "src")
import numpy as np, torch
from pysph.base.kernels import CubicSpline
from tensorsph.droplet_impact import cubic_spline_kernel, cubic_spline_gradient
import inspect
print(inspect.getsource(cubic_spline_gradient)[:1200])

h = 1.3e-2
k = CubicSpline(dim=2)
for r in (0.0, 0.3 * h, 1.0 * h, 1.7 * h):
    xij = np.array([r, 0.0, 0.0])
    Wp = k.kernel(xij=np.array([r, 0.0, 0.0]), rij=r, h=h)
    gp = k.gradient(xij=np.array([r, 0.0, 0.0]), rij=r, h=h)
    d = torch.tensor([[r, 0.0]])
    Wm = float(cubic_spline_kernel(torch.tensor(r), h))
    gm = cubic_spline_gradient(d, h).numpy()
    print(f"r/h={r/h:.2f} W pysph={Wp:.6e} ours={Wm:.6e} | grad pysph={gp} ours={gm}")