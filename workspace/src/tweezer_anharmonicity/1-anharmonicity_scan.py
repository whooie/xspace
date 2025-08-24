from pathlib import Path
import sys
import numpy as np
import whooie.phys as phys
import whooie.pyplotdefs as pd

outdir = Path("output")

data = np.load(str(outdir.joinpath("he-tweezer-anharm.npz")))
depth = data["depth"]
anharm = data["anharm"]
waist = data["waist"][0]
mass = data["mass"][0]
lscale = data["lscale"][0]

anharm_valid = anharm[np.isfinite(anharm)]
ymin = anharm_valid.min()
ymax = anharm_valid.max()
yrange = ymax - ymin

(
    pd.Plotter()
    .plot(depth, anharm)
    .ggrid()
    .set_ylim(ymin - yrange / 20, ymax + yrange / 20)
    .set_xlabel("Trap depth [MHz]")
    .set_ylabel("$(E_1 - E_0) / (E_2 - E_1) - 1$")
    .set_title(f"waist = {waist:g} μm")
    .savefig(outdir.joinpath("he-tweezer_anharm-scan.png"))
    .close()
)

