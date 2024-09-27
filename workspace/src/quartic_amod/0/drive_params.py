from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

FS = pd.pp.rcParams["figure.figsize"]

outdir = Path("output")
da = np.array([0.025, 0.05, 0.1, 0.15, 0.20])
p2max = np.array([0.999710, 0.999407, 0.997759, 0.994671, 0.989239])
reldet = np.array([1.000136, 0.999361, 0.997619, 0.995895, 0.993223])
rabi_freq = np.array([1.468, 2.935, 5.860, 8.835, 11.667]) / 2

pd.Plotter.new(
    nrows=3,
    sharex=True,
    figsize=[FS[0], 1.15 * FS[1]],
    as_plotarray=True,
) \
    [0] \
    .semilogy(da, 1 - p2max, marker="o", linestyle="-", color="C0") \
    .ggrid() \
    .set_ylabel("$1 - P_2^\\mathregular{max}$", fontsize="small") \
    [1] \
    .plot(da, reldet - 1, marker="o", linestyle="-", color="k") \
    .ggrid() \
    .set_ylabel("$\\delta f$", fontsize="small") \
    [2] \
    .plot(da, rabi_freq, marker="o", linestyle="-", color="C1") \
    .ggrid() \
    .set_ylabel("$\\Omega_\\mathregular{eff}$ [kHz]", fontsize="small") \
    .set_xlabel("$\\delta a$") \
    .tight_layout(h_pad=0.5) \
    .savefig(outdir.joinpath("quartic_amod_drive_params.png")) \
    .close()

