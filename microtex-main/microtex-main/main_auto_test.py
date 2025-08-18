import microtextures
import os
import numpy as np
import time
import PyAnsys.PyFluent_main as pyfluent_main
import PyAnsys.SC_cmd as sc_cmd
import PyAnsys.boundarylayerthickness as blt
from tomlkit import parse

def main(depth, width, gap):
    tick = time.time()

    with open('config.toml', 'r') as file:
        toml_string = file.read()
        config = parse(toml_string)

    fp_testbed = (config["filepaths"]["models"])

    # change working directory
    os.chdir(fp_testbed)

    # create model
    model = microtextures.Scallop(depth, width, gap, 0).texture_disc()
    microtextures.CQModel().export_STEP(model, 8)

    # store a copy of the model with identifiers in name
    microtextures.CQModel().export_STEP(model, r'_' + str(depth) + r'_' + str(width) + r'_' + str(gap))

    tock = time.time()
    print("Model creation time taken: ", tock - tick)

    # run SC script
    tick = time.time()
    sc_cmd.execute()
    tock = time.time()
    print("SC script execution time taken: ", tock - tick)

    # run Fluent meshing and simulations
    tick = time.time()
    pyfluent_main.main()
    tock = time.time()
    print("Fluent meshing and simulation time taken: ", tock - tick)

    # read resultant csv and calculate boundary layer thickness
    tick = time.time()
    y_value = blt.execute()
    tock = time.time()
    print("Boundary layer thickness calculation time taken: ", tock - tick)
    
    return y_value

if __name__=='__main__':
    boundary_layer_thickness = main(0.05,0.05,0.05)
    print(boundary_layer_thickness)