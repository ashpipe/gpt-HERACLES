from gpt import GPT
from distgen import Generator
import numpy as np


def run(MTE=200.0, sol2_current=3.0, spotSize=0.8):

    # generate particles
    gen = Generator("../beamline/uniform.in.yaml", verbose=0)
    gen["total_charge:value"] = 1e-3  # 3 uA
    gen["r_dist:max_r:value"] = spotSize

    gen["start:MTE:value"] = MTE

    gen.run()

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
    # set scan range
    sol2_arr = np.linspace(0, 3.7, num=3)
    MTE_arr = np.linspace(27, 500, num=3)
    # spotSize_arr = np.linspace(0.3, 3, num=3)
    sig_3_arr = np.zeros((len(MTE_arr), len(sol2_arr)))  # 3rd screen
    sig_d_arr = np.zeros((len(MTE_arr), len(sol2_arr)))  # dump
    rmax_3_arr = np.zeros((len(MTE_arr), len(sol2_arr)))
    rmax_d_arr = np.zeros((len(MTE_arr), len(sol2_arr)))


if __name__ == "__main__":
    pass
