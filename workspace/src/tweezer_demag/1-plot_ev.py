from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.fft as fft
import whooie.pyplotdefs as pd

@dataclass
class WfStats:
    t: np.ndarray[float, 1]
    max: np.ndarray[float, 1]
    mean: np.ndarray[float, 1]
    std_p: np.ndarray[float, 1]
    std_m: np.ndarray[float, 1]
    sh: float

    @staticmethod
    def from_file(infile: Path) -> "WfStats":
        data = np.load(str(infile))
        t = data["t"]
        x = data["x"]
        dx = x[1] - x[0]
        sh = data["sh"][0]
        p = (abs(data["q"]) ** 2).mean(axis=0)
        pnorm = np.trapezoid(p, dx=dx, axis=1)
        p = np.array([pk / nk for (pk, nk) in zip(p, pnorm)])
        max = x[np.argmax(p, axis=1)]
        mean = np.trapezoid(p * x, dx=dx, axis=1)
        xdiff = np.array([x - m for m in mean])
        norm_p = np.trapezoid((xdiff < 0) * p, dx=dx, axis=1)
        renorm_p = np.array([(pk / npk) if abs(npk) > 1e-12 else pk for (pk, npk) in zip(p, norm_p)])
        std_p = np.sqrt(np.trapezoid(xdiff ** 2 * (xdiff < 0) * renorm_p, dx=dx, axis=1))
        norm_m = np.trapezoid((xdiff > 0) * p, dx=dx, axis=1)
        renorm_m = np.array([(pk / nmk) if abs(nmk) > 1e-12 else pk for (pk, nmk) in zip(p, norm_m)])
        std_m = np.sqrt(np.trapezoid(xdiff ** 2 * (xdiff > 0) * renorm_m, dx=dx, axis=1))
        return WfStats(t, max, mean, std_p, std_m, sh)

    @staticmethod
    def k_from_file(infile: Path) -> "WfStats":
        data = np.load(str(infile))
        t = data["t"]
        x = data["x"]
        dx = x[1] - x[0]
        k = fft.fftshift(fft.fftfreq(x.shape[0], d=dx))
        dk = k[1] - k[0]
        sh = 0.0
        p = fft.fftshift((abs(fft.fft(data["q"], axis=2)) * 2).mean(axis=0), axes=1)
        pnorm = np.trapezoid(p, dx=dk, axis=1)
        p = np.array([pk / nk for (pk, nk) in zip(p, pnorm)])
        max = k[np.argmax(p, axis=1)]
        mean = np.trapezoid(p * k, dx=dk, axis=1)
        kdiff = np.array([k - m for m in mean])
        norm_p = np.trapezoid((kdiff < 0) * p, dx=dk, axis=1)
        renorm_p = np.array([(pk / npk) if abs(npk) > 1e-12 else pk for (pk, npk) in zip(p, norm_p)])
        std_p = np.sqrt(np.trapezoid(kdiff ** 2 * (kdiff < 0) * renorm_p, dx=dk, axis=1))
        norm_m = np.trapezoid((kdiff > 0) * p, dx=dk, axis=1)
        renorm_m = np.array([(pk / nmk) if abs(nmk) > 1e-12 else pk for (pk, nmk) in zip(p, norm_m)])
        std_m = np.sqrt(np.trapezoid(kdiff ** 2 * (kdiff > 0) * renorm_m, dx=dk, axis=1))
        return WfStats(t, max, mean, std_p, std_m, sh)


indir = Path("output").joinpath("harmonic-demag_z=1.25")
# indir = Path("output").joinpath("harmonic-demag_z=2")
# indir = Path("output").joinpath("harmonic-demag_z=10")
# indir = Path("output").joinpath("harmonic-demag_z=100")
# indir = Path("output").joinpath("tweezer-demag_z=100")
# sh = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]) # μm
sh = np.array([1.0, 5.0, 10.0, 15.0, 20.0]) # μm

# indir = Path("output").joinpath("harmonic")
# indir = Path("output").joinpath("tweezer")
# sh = np.array([1.0, 5.0, 10.0, 15.0, 20.0]) # μm

infiles = [indir.joinpath(f"sites_ev_sh={sh_k:.1f}.npz") for sh_k in sh]
data = [WfStats.from_file(infile) for infile in infiles]
kdata = [WfStats.k_from_file(infile) for infile in infiles]

fs = pd.pp.rcParams["figure.figsize"]
P = pd.Plotter.new(
    nrows=2, sharex=True, figsize=[fs[0], 1.5 * fs[1]], as_plotarray=True)
for (k, (wf, kwf)) in enumerate(zip(data, kdata)):
    P[0].fill_between(
        wf.t, wf.mean - wf.std_m, wf.mean + wf.std_p,
        linewidth=0.0, color=f"C{k % 10}", alpha=0.25,
    )
    P[0].plot(
        wf.t, wf.mean,
        linestyle="-", marker="", color=f"C{k % 10}",
    )
    # P[0].plot(
    #     wf.t, wf.max,
    #     linestyle="--", marker="", color=f"C{k % 10}",
    # )
    P[1].fill_between(
        kwf.t, kwf.mean - kwf.std_m, kwf.mean + kwf.std_p,
        linewidth=0.0, color=f"C{k % 10}", alpha=0.25,
    )
    P[1].plot(
        kwf.t, kwf.mean,
        linestyle="-", marker="", color=f"C{k % 10}",
    )
    # P[1].plot(
    #     kwf.t, kwf.max,
    #     linestyle="--", marker="", color=f"C{k % 10}",
    # )
(
    P
    [0]
    .ggrid()
    .set_ylabel("$x$ [nat.]")
    [1]
    .ggrid()
    .set_ylabel("$k$ [arb.]")
    .set_xlabel("Time [nat.]")
    .savefig(indir.joinpath("sites_ev.png"))
)

