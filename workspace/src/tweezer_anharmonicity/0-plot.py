from pathlib import Path
import sys
import numpy as np
import whooie.phys as phys
import whooie.pyplotdefs as pd

outdir = Path("output")

data = np.load(str(outdir.joinpath("he-tweezer.npz")))
x = data["x"]
v = data["v"]
e = data["e"]
enat = data["enat"][0]
a = data["a"][0]
wf = data["wf"]

ediff = e[1:] - e[:-1]
De = e.max() - e.min()

w = np.sqrt(4 * 1e6 * phys.h / 5e-27 / 0.5e-6 ** 2)
print(w / 2 / np.pi / 1e3, file=sys.stderr)
print(w * phys.hbar / enat, file=sys.stderr)
x_approx = np.linspace(-0.5e-6, 0.5e-6, 1001)
v_approx = 0.5 * 5e-27 * w ** 2 * x_approx ** 2

P = pd.Plotter()
# P.plot(x_approx / a, v_approx / enat, color="0.5").plot(x, v, color="k")
P.plot(x, v * enat / phys.h / 1e3, color="k")
for (k, (ek, wfk)) in enumerate(zip(e, wf)):
    (
        P
        .axhline(ek * enat / phys.h / 1e3, linestyle="--", color="0.5")
        .plot(x, (ek + 15 * wfk) * enat / phys.h / 1e3, color="C0")
    )
(
    P
    .ggrid()
    # .set_ylim(e.min() - De / 10, e.max() + De / 10)
    .set_xlabel("$x$ [μm]")
    .set_ylabel("Energy [kHz]")
    .savefig(outdir.joinpath("he-tweezer_wfs.png"))
    .savefig(outdir.joinpath("he-tweezer_wfs.pdf"))
    .close()
)

FS = pd.pp.rcParams["figure.figsize"]
(
    pd.Plotter.new(
        nrows=2,
        sharex=True,
        # figsize=[FS[0], FS[1]],
        as_plotarray=True)
    [0]
    .plot(e * enat / phys.h / 1e3, marker="o", linestyle="", color="C0")
    .ggrid()
    .set_ylabel("Energy [kHz]", fontsize="small")
    .set_title(
        f"$E_1 - E_0 = {(e[1] - e[0]) * enat / phys.h / 1e3:g}$ kHz\n"
        f"$E_2 - E_1 = {(e[2] - e[1]) * enat / phys.h / 1e3:g}$ kHz"
    )
    [1]
    .plot(
        0.5 + np.arange(ediff.shape[0]),
        ediff * enat / phys.h / 1e3,
        marker="o", linestyle="", color="C1",
    )
    .ggrid()
    .set_ylabel("Diff. [kHz]", fontsize="small")
    .set_xlabel("$\\nu$")
    .tight_layout(h_pad=0.5)
    .savefig(outdir.joinpath("he-tweezer_spectrum.png"))
    .close()
)

