import numpy as np
import whooie.pyplotdefs as pd

data = np.load("output/quartic_shooting.npz")
e = data["e"]
d = data["d"]

pd.Plotter().plot(e, d).show()

