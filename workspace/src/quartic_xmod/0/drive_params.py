from pathlib import Path
import numpy as np
import whooie.pyplotdefs as pd

FS = pd.pp.rcParams["figure.figsize"]

outdir = Path("output")
dx = np.array([0.001, 0.0025, 0.005, 0.0075, 0.01]) * 1e3
p1max = np.array([0.999869, 0.999305, 0.996880, 0.993217, 0.989240])
reldet = np.array([0.999024, 0.998313, 0.997243, 0.996000, 0.994613])
rabi_freq = np.array([0.387, 0.996, 1.930, 2.867, 3.851])

pd.Plotter.new(
    nrows=3,
    sharex=True,
    figsize=[FS[0], 1.15 * FS[1]],
    as_plotarray=True,
) \
    [0] \
    .semilogy(dx, 1 - p1max, marker="o", linestyle="-", color="C0") \
    .ggrid() \
    .set_ylabel("$1 - P_1^\\mathregular{max}$", fontsize="small") \
    [1] \
    .plot(dx, reldet - 1, marker="o", linestyle="-", color="k") \
    .ggrid() \
    .set_ylabel("$\\delta f$", fontsize="small") \
    [2] \
    .plot(dx, rabi_freq, marker="o", linestyle="-", color="C1") \
    .ggrid() \
    .set_ylabel("$\\Omega_\\mathregular{eff}$ [kHz]", fontsize="small") \
    .set_xlabel("$\\delta x$ [nm]") \
    .tight_layout(h_pad=0.5) \
    .savefig(outdir.joinpath("quartic_xmod_drive_params.png")) \
    .close()

