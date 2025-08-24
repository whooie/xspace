from pathlib import Path
import sys
import numpy as np
from scipy.special import erf
import whooie.phys as phys
import whooie.pyplotdefs as pd

outdir = Path("output")
data = np.load(str(outdir.joinpath("tweezer_amod.npz")))
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

de1 = (e1 - e0) * enat
de2 = (e2 - e0) * enat

dw1 = de1 / phys.hbar
dw2 = de2 / phys.hbar

tau1 = 2 * np.pi / dw1 * 100
tmax1 = 5 * tau1
t1 = np.linspace(0.0, tmax1, 1000)
dph1 = np.pi * (erf((t1 - 2.5 * tau1) / tau1) + 1)

tau2 = 2 * np.pi / dw2 * 100
tmax2 = 5 * tau2
t2 = np.linspace(0.0, tmax2, 1000)
dph2 = np.pi * (erf((t2 - 2.5 * tau2) / tau2) + 1)

(
    pd.Plotter.new(figsize=[2.8, 1.5])
    .plot(t1 * 1e6, dph1, c="C1", label="$\\Delta \\varphi_1$")
    .plot(t2 * 1e6, dph2, c="C2", label="$\\Delta \\varphi_2$")
    .ggrid()
    .legend(fontsize="x-small", loc="lower right")
    .set_xlabel("Time [μs]")
    .set_ylabel("Acc. phase [rad]")
    .set_yticks([0.0, np.pi, 2 * np.pi], ["$0$", "$\\pi$", "$2 \\pi$"])
    .savefig(outdir.joinpath("tweezer-phase-acc.pdf"))
    .close()
)

