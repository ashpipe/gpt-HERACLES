from gpt import GPT
from distgen import Generator
import numpy as np
import h5py

repeat_count = 20000


class data:
    def __init__(self, fn="data.h5") -> None:
        self.fn = fn

    def read(self):
        f = h5py.File(self.fn, "r")
        dataset0 = f["dataset_0"][:]
        f.close
        return dataset0

    def write(self, arr: np.ndarray):

        with h5py.File(self.fn, "w") as f:
            dset = f.create_dataset("dataset_0", data=arr)


class sim:
    @staticmethod
    def gen(MTE: float = 200.0, spotSize: float = 0.8) -> Generator:

        # generate particles
        gen = Generator("../beamline/uniform.in.yaml", verbose=0)
        gen["total_charge:value"] = 1e-3  # 3 uA
        gen["r_dist:max_r:value"] = float(spotSize)

        gen["start:MTE:value"] = float(MTE)

        gen.run()

        return gen

    @staticmethod
    def gpt(gen: Generator, sol2_current: float = 3.0) -> list:

        # run GPT
        G = GPT(
            input_file="../beamline/beamline.in",
            initial_particles=gen.particles,
            verbose=False,
        )
        G.set_variables({"sol02_current": sol2_current})
        G.run()

        # results (unit in cm)
        sig_3 = G.stat("sigma_x", "screen")[2] * 100  # sigma_x 3rd screen
        sig_d = G.stat("sigma_x", "screen")[3] * 100  # sigma_x dump
        rmax_3 = G.stat("max_r", "screen")[2] * 100
        rmax_d = G.stat("max_r", "screen")[3] * 100

        return [sig_3, sig_d, rmax_3, rmax_d]


def main():

    xdata_file = data(fn="xdata.h5")
    ydata_file = data(fn="ydata.h5")
    xdata0 = []
    ydata0 = []

    for ii in range(repeat_count):
        sol2 = np.random.default_rng().uniform(0, 6.0, size=1)[0]
        MTE = np.random.default_rng().uniform(27, 1500, size=1)[0]
        spotSize = np.random.default_rng().uniform(0.3, 1.5, size=1)[0]

        gen = sim.gen(MTE=MTE, spotSize=spotSize)
        xdata_entry = [MTE, spotSize, sol2]
        ydata_entry = sim.gpt(gen, sol2_current=sol2)
        xdata0.append(xdata_entry)
        ydata0.append(ydata_entry)

        if ii % 200 == 1:
            xdata_file.write(np.array(xdata0))
            ydata_file.write(np.array(ydata0))


if __name__ == "__main__":
    main()
