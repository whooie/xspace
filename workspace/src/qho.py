from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

outdir = Path("output")

data = np.load(str(outdir.joinpath("qho.npz")))
x = data["x"]
v = data["v"]
e = data["e"]
wf = data["wf"]

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
    .savefig(outdir.joinpath("qho_wfs.png"))
    .close()
)

(
    pd.Plotter.new(nrows=2, sharex=True, as_plotarray=True)
    [0]
    .plot(
        1 + 2 * np.arange(e.shape[0]),
        marker="D", linestyle="", color="k",
        label="Analytical",
    )
    .plot(
        e,
        marker="o", linestyle="", color="C0",
        label="Computed",
    )
    .ggrid()
    .legend(fontsize="xx-small")
    .set_ylabel("Energy [nat.]")
    [1]
    .semilogy(
        abs(e - (1 + 2 * np.arange(e.shape[0]))),
        marker="o", linestyle="", color="C1",
    )
    .ggrid()
    .set_xlabel("$\\nu$")
    .set_ylabel("Abs. error")
    .tight_layout(h_pad=0.5)
    .savefig(outdir.joinpath("qho_spectrum.png"))
    .close()
)

