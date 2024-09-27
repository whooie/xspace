from pathlib import Path
import numpy as np
import whooie.phys as phys
import whooie.pyplotdefs as pd

outdir = Path("output")

data = np.load(str(outdir.joinpath("quartic.npz")))
x = data["x"]
v = data["v"]
e = data["e"]
enat = data["enat"][0]
a = data["a"][0]
wf = data["wf"]
tps = data["tps"]

print(f"a = {a * enat / phys.h / 1e3:.3} kHz / μm⁴")

ediff = e[1:] - e[:-1]
De = e.max() - e.min()

P = pd.Plotter()
P.plot(x, v, color="k")
for (k, (ek, wfk)) in enumerate(zip(e, wf)):
    (
        P
        .axhline(ek, linestyle="--", color="0.5")
        .plot(x, ek + wfk, color="C0")
    )
(
    P
    .ggrid()
    .set_ylim(e.min() - De / 10, e.max() + De / 10)
    .set_xlabel("$x$")
    .set_ylabel("Energy [nat.]")
    .savefig(outdir.joinpath("quartic_wfs.png"))
    .close()
)

FS = pd.pp.rcParams["figure.figsize"]
(
    pd.Plotter.new(
        nrows=3,
        sharex=True,
        figsize=[FS[0], 1.25 * FS[1]],
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
    [2]
    .plot(tps, marker="o", linestyle="", color="k")
    .ggrid()
    # .set_ylim(bottom=0)
    .set_ylabel("Turning pt. [μm]", fontsize="small")
    .set_xlabel("$\\nu$")
    .tight_layout(h_pad=0.5)
    .savefig(outdir.joinpath("quartic_spectrum.png"))
    .close()
)

