from itertools import product
from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd
from whooie.iters import Iter

# indir = Path("output").joinpath("harmonic")
indir = Path("output").joinpath("harmonic-demag_z=1.25")
# indir = Path("output").joinpath("harmonic-demag_z=2")
# indir = Path("output").joinpath("harmonic-demag_z=10")
# indir = Path("output").joinpath("harmonic-demag_z=100")
fname = lambda sh: indir.joinpath(f"sites_ev_sh={sh:.1f}.npz")

def do_avg_dist(
    x1: np.ndarray[float, 1],
    p1: np.ndarray[float, 1],
    x2: np.ndarray[float, 1],
    p2: np.ndarray[float, 1],
) -> (float, float):
    dx1 = x1[1] - x1[0]
    dx2 = x2[1] - x2[0]
    X1, X2 = np.meshgrid(x1, x2)
    pp = np.outer(p2, p1)
    m1 = np.trapezoid(np.trapezoid((X2 - X1) * pp, dx=dx1), dx=dx2)
    m2 = np.trapezoid(np.trapezoid((X2 - X1) ** 2 * pp, dx=dx1), dx=dx2)
    mean = m1
    std = np.sqrt(m2 - m1 ** 2)
    return (mean, std)

def avg_dist(
    sh1: float,
    sh2: float,
) -> (np.ndarray[float, 1], np.ndarray[float, 1], np.ndarray[float, 1]):
    data1 = np.load(str(fname(sh1)))
    t1 = data1["t"]
    x1 = data1["x"]
    p1 = (abs(data1["q"]) ** 2).mean(axis=0)[::10]
    data2 = np.load(str(fname(sh2)))
    t2 = data2["t"]
    x2 = data2["x"]
    p2 = (abs(data2["q"]) ** 2).mean(axis=0)[::10]
    assert np.all(t1 == t2)
    dist = np.array([
        do_avg_dist(x1, pt1, x2, pt2) for (pt1, pt2) in zip(p1, p2)])
    return (t1[::10], *dist.T)

cache_file = indir.joinpath("overlaps_cache.npz")
try:
    data = np.load(str(cache_file))
    pairs = data["pairs"]
    dists = data["dists"]
    print("loaded from cache file")
except:
    print("could not load cache file")
    sh = np.array([1.0, 5.0, 10.0, 15.0, 20.0]) # μm
    # sh = np.array([1.0, 5.0]) # μm
    m = len(sh)
    pairs_iter = (
        Iter(sh)
        .enumerate()
        .flat_map(
            lambda k_sh: (
                Iter(sh)
                .take(k_sh[0])
                .map(lambda sh2: (sh2, k_sh[1]))
            )
        )
    )
    pairs = np.array(pairs_iter.collect_list())
    dists = np.array([avg_dist(sh1, sh2) for (sh1, sh2) in pairs])
    np.savez(str(cache_file), pairs=pairs, dists=dists)

P = pd.Plotter()
for (k, (pair, dist)) in Iter(pairs).zip(Iter(dists)).enumerate():
    (sh1, sh2) = pair
    (t, d, e) = dist
    P.fill_between(
        t, d - e, d + e,
        linewidth=0.0, color=f"C{k % 10}", alpha=0.15,
    )
    P.plot(
        t, d,
        linestyle="-", color=f"C{k % 10}",
        label=f"$x_0 = {sh1:.1f}$ μm; $x_0' = {sh2:.1f}$ μm",
    )
(
    P
    .ggrid()
    .legend(
        fontsize="xx-small",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
    )
    .set_xlabel("Time [nat.]")
    .set_ylabel("$\\langle x' - x\\rangle$ [nat.]")
    .savefig(indir.joinpath("distances.png"))
    .close()
)

