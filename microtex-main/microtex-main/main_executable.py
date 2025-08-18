import microtextures
import os
import numpy as np
import time



def main():
    tick = time.time()
    # change working directory
    os.chdir(r"D:\Work\Summer Placement Testbed")

    # https://stackoverflow.com/questions/16552508/python-loops-for-simultaneous-operation-two-or-possibly-more
    # to enable customisable ranges in geometry
    
    #for i, angle in zip(range(40,43), np.linspace(0, 90, 3)):
    #
    #    model = microtextures.Scallop(0.1, 0.15, 0.2, angle).texture_disc()
    #    microtextures.CQModel().export_STEP(model, i)
    model = microtextures.Scallop(0.1, 0.15, 0.2, 90).texture_disc()
    microtextures.CQModel(0,0,0).export_STEP(model, 8)

    tock = time.time()
    print("Model creation time taken: ", tock - tick)

    # SpaceClaim installation Directory
    # Made a variable in case we want to give the user the choice to change the installation directory later
    filepath = r"D:\Program Files\ANSYS Inc\ANSYS Student\v232\scdm"


    os.chdir(filepath)
    os.system('cmd /c spaceclaim.exe /RunScript="D:\Work\Summer Placement Testbed\model42_fluiddomain_topflush.py" /Headless=True /Splash=False /Welcome=False /ExitAfterScript=True"')

if __name__=='__main__':
    main()