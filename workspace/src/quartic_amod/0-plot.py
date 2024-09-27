from pathlib import Path
import sys
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output")
data = np.load(str(outdir.joinpath("quartic_amod.npz")))
mass = data["mass"][0]
anat = data["anat"][0]
enat = data["enat"][0]
tnat = data["tnat"][0]
x = data["x"]
v = data["v"]
q0 = data["q0"]
e0 = data["e0"][0]
q1 = data["q1"]
e1 = data["e1"][0]
q2 = data["q2"]
e2 = data["e2"][0]
fdiff = data["fdiff"][0]
ampl = data["ampl"][0]
freq = data["freq"][0]
t = data["t"]
vt = data["vt"]
q = data["q"]
a0 = data["a0"]
a1 = data["a1"]
a2 = data["a2"]

P0 = abs(a0) ** 2
P1 = abs(a1) ** 2
P2 = abs(a2) ** 2
P012 = P0 + P1 + P2

print(P2.max(), file=sys.stderr)

t *= tnat * 1e6
k0 = P2.argmax()
t0 = t[k0]
rabi_freq = 0.5 / t0 * 1e3

title = (
    f"$\\delta a = {ampl:.3f}$; "
    f"$\\delta f = {freq / fdiff - 1:.6f}$; "
    f"$\\Omega_\\mathregular{{eff}} = {rabi_freq:.3f}$ kHz"
)

pd.Plotter() \
    .colorplot(x, t, abs(q) ** 2) \
    .colorbar() \
    .ggrid().grid(False, which="both") \
    .set_xlabel("$x$ [μm]") \
    .set_ylabel("$t$ [μs]") \
    .set_title(title) \
    .savefig(outdir.joinpath(f"quartic_amod_wf_da={ampl:.3f}.png")) \
    .close()

pd.Plotter() \
    .plot(t, P012, color="k", label="$P_0 + P_1 + P_2$") \
    .plot(t, P0, label="$P_0$") \
    .plot(t, P1, label="$P_1$") \
    .plot(t, P2, label="$P_2$") \
    .ggrid() \
    .legend(fontsize="xx-small") \
    .set_xlabel("$t$ [μs]") \
    .set_ylabel("Probability") \
    .set_title(title) \
    .savefig(outdir.joinpath(f"quartic_amod_probs_da={ampl:.3f}.png")) \
    .close()

# pd.show()

