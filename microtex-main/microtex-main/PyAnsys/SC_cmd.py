# Purpose: To run SpaceClaim script from command line, on a headless instance of SpaceClaim
# SpaceClaim script is used to create the fluid domain from a model, it can be run in GUI for easier debugging
import os
import sys
from tomlkit import parse
import time

def execute():

    #append parent directory to sys.path so that config.toml can be accessed
    sys.path.append("..")

    with open('config.toml', 'r') as file:
        toml_string = file.read()
        config = parse(toml_string)

    # SpaceClaim installation Directory
    # Made a variable in case we want to give the user the choice to change the installation directory later
    filepath = (config["filepaths"]["ansys_scdm"])

    # finds the directory this script is in and appends SC_script.py to it
    filepath_SC_script = os.path.join(os.path.dirname(__file__), 'SC_script_25_enclosure.py')

    cmd = r'cmd /c spaceclaim.exe /RunScript="' + filepath_SC_script + r'" /Headless=True /Splash=False /Welcome=False /ExitAfterScript=True'
    #cmd = r'cmd /c spaceclaim.exe /RunScript="E:\Summer Placement\microtex\PyAnsys\SC_test.py" /Headless=False /Splash=False /Welcome=False /ExitAfterScript=False'

    #cmd = r'cmd /c D:\"Program Files"\"ANSYS Inc"\"ANSYS Student"\v232\scdm\spaceclaim.exe /RunScript="D:\Work\microtex\PyAnsys\SC_script.py" /Headless=True /Splash=False /Welcome=False /ExitAfterScript=True'
    #this works too but it's tricky to set the path in a way CMD will accept
    
    os.chdir(filepath)
    os.system(cmd)
    # must use /c instead of /k to close the prompt after execution if running terminal in vscode
    # we may want to keep SCDM open to accept more scripts without having to open a new instance
    # if so, build in a function to close SCDM after all scripts have been run

if __name__=='__main__':
    tick = time.time()
    execute()
    tock = time.time()
    print("SC script execution time taken: ", tock - tick)