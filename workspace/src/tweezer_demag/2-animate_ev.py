from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numpy.fft as fft
import whooie.pyplotdefs as pd
pd.pp.rcParams["font.size"] = 5.0
from matplotlib.animation import FuncAnimation

@dataclass
class WfStats:
    t: np.ndarray[float, 1]
    x: np.ndarray[float, 1]
    p: np.ndarray[float, 2]
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
        return WfStats(t, x, p, max, mean, std_p, std_m, sh)

@dataclass
class WfStatsK:
    t: np.ndarray[float, 1]
    k: np.ndarray[float, 1]
    p: np.ndarray[float, 2]
    max: np.ndarray[float, 1]
    mean: np.ndarray[float, 1]
    std_p: np.ndarray[float, 1]
    std_m: np.ndarray[float, 1]
    sh: float

    @staticmethod
    def from_file(infile: Path) -> "WfStatsK":
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
        return WfStatsK(t, k, p, max, mean, std_p, std_m, sh)

sh = 20.0
z = 1.25
infile = (
    Path("output")
    # .joinpath("harmonic")
    .joinpath(f"harmonic-demag_z={z:g}")
    .joinpath(f"sites_ev_sh={sh:.1f}.npz")
)
xdata = WfStats.from_file(infile)
kdata = WfStatsK.from_file(infile)

xpmax = xdata.p.max()
kpmax = kdata.p.max()
nframes = xdata.t.shape[0]
w = int(np.log10(nframes)) + 1

P = pd.Plotter.new(nrows=2, log=False, chain=False, as_plotarray=True)
P[0].text_ax(0.02, 0.98, f"$x_0 = {sh:.1f}$ μm", ha="left", va="top", fontsize="small")
P[0].set_title(f"$t = {0.0:.3f}$ [nat.]")

xmax = 600.0
# P[0].set_xlabel("$x$ [nat.]")
# P[0].set_xlim(xdata.x.min(), xdata.x.max())
# P[0].set_xlim(-800.0, 800.0)
P[0].set_xlim(-xmax / z * 1.25, xmax)
# P[0].set_xlim(-1100.0, 1100.0)
# P[0].set_ylabel("Prob. dens.")
P[0].set_ylim(-0.05 * xpmax, 1.05 * xpmax)

# P[1].set_xlabel("$k$ [nat.]")
P[1].set_xlim(kdata.k.min(), kdata.k.max())
# P[1].set_ylabel("Prob. dens.")
P[1].set_ylim(-0.05 * kpmax, 1.05 * kpmax)

xmean = P[0].axvline(np.float64("nan"), linestyle="-", linewidth=0.4, color="k")
xmean_m = P[0].axvline(np.float64("nan"), linestyle="--", linewidth=0.4, color="k")
xmean_p = P[0].axvline(np.float64("nan"), linestyle="--", linewidth=0.4, color="k")
xline, = P[0].plot([], [], linestyle="-", color="C0", label="$x$-space\n(nat. units)")
P[0].legend(
    fontsize="xx-small",
    frameon=False,
    loc="upper right",
    framealpha=1.0,
)

kmean = P[1].axvline(np.float64("nan"), linestyle="-", linewidth=0.4, color="k")
kmean_m = P[1].axvline(np.float64("nan"), linestyle="--", linewidth=0.4, color="k")
kmean_p = P[1].axvline(np.float64("nan"), linestyle="--", linewidth=0.4, color="k")
kline, = P[1].plot([], [], linestyle="-", color="C3", label="$k$-space\n(nat. units)")
P[1].legend(
    fontsize="xx-small",
    frameon=False,
    loc="upper right",
    framealpha=1.0,
)

def init():
    xline.set_xdata(xdata.x)
    xline.set_ydata(xdata.x.shape[0] * [np.float64("nan")])
    kline.set_xdata(kdata.k)
    kline.set_ydata(kdata.k.shape[0] * [np.float64("nan")])
    return (xline, kline)

def animate(j: int):
    P[0].set_title(f"$t = {xdata.t[j]:.3f}$ [nat.]")
    xline.set_ydata(xdata.p[j, :])
    xmean.set_xdata(2 * [xdata.mean[j]])
    xmean_m.set_xdata(2 * [xdata.mean[j] - xdata.std_m[j]])
    xmean_p.set_xdata(2 * [xdata.mean[j] + xdata.std_p[j]])
    kline.set_ydata(kdata.p[j, :])
    kmean.set_xdata(2 * [kdata.mean[j]])
    kmean_m.set_xdata(2 * [kdata.mean[j] - kdata.std_m[j]])
    kmean_p.set_xdata(2 * [kdata.mean[j] + kdata.std_p[j]])
    return (xline, kline)

animation = FuncAnimation(
    P.fig,
    animate,
    init_func=init,
    frames=xdata.p.shape[0],
    interval=50,
    blit=False,
)

def save_progress(cur_frame: int, total_frames: int):
    print(
        f"  frame = {{:{w}}} / {{:{w}}}".format(cur_frame, total_frames),
        end="\r",
        flush=True,
    )

animation.save(
    str(infile.with_stem(infile.stem + "_anim").with_suffix(".mkv")),
    writer="ffmpeg",
    fps=20,
    progress_callback=save_progress,
)

print(f"  frame = {{:{w}}} / {{:{w}}}".format(nframes, nframes), flush=True)

