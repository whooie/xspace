import numpy as np
import numpy.fft as fft
import whooie.pyplotdefs as pd

data = np.load("qho.npz")
x = data["x"]
v = data["v"]
e = float(data["e"][0])
wf = data["wf"]
v_shifted = data["v_shifted"]
wf_shifted = data["wf_shifted"]
t = data["t"]
q = data["q"]

X, T = np.meshgrid(x, t)

r = v.max() - v.min()

pd.Plotter() \
    .axhline(e, linestyle="--", color="0.35") \
    .plot(x, v, color="k") \
    .plot(x, e + wf * r / 10, color="C0")
pd.Plotter() \
    .axhline(e, linestyle="--", color="0.35") \
    .plot(x, v_shifted, color="k") \
    .plot(x, e + wf_shifted * r / 10, color="C1")
pd.Plotter() \
    .colorplot(x, t, abs(q) ** 2, cmap="vibrant") \
    .colorbar()
pd.Plotter.new_3d() \
    .plot_surface(X, T, abs(q) ** 2, cmap="vibrant") \
    .set_zlim(0, wf.max())

pd.show()

