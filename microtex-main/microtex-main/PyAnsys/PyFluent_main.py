import ansys.fluent.core as pyfluent
import time

def main():

    meshing_session = pyfluent.launch_fluent(precision="double", processor_count=4, mode="meshing", show_gui=False, start_transcript=False)
    # set start_tracript to False for now? product_version="23.2.0" is also an argument that can be passed
    workflow = meshing_session.workflow
    meshing = meshing_session.meshing

    # Initialise workflow for meshing
    workflow.InitializeWorkflow(WorkflowType=r'Watertight Geometry')
    meshing.GlobalSettings.LengthUnit.set_state(r'um')
    meshing.GlobalSettings.AreaUnit.set_state(r'um^2')
    meshing.GlobalSettings.VolumeUnit.set_state(r'um^3')

    # Import geometry
    workflow.TaskObject['Import Geometry'].Arguments.set_state({r'FileName': r'E:\Summer Placement\Summer Placement Testbed\automationtest.scdoc',
                                                             "LengthUnit": "um",})
    workflow.TaskObject['Import Geometry'].Execute()

    # Add local sizing (or not)
    workflow.TaskObject['Add Local Sizing'].AddChildAndUpdate()

    # Generate Surface Mesh
    workflow.TaskObject['Generate the Surface Mesh'].Execute()

    # Describe geometry
    workflow.TaskObject['Describe Geometry'].UpdateChildTasks(SetupTypeChanged=False)
    workflow.TaskObject['Describe Geometry'].Arguments.set_state({r'SetupType': r'The geometry consists of only fluid regions with no voids',})
    workflow.TaskObject['Describe Geometry'].UpdateChildTasks(SetupTypeChanged=True)
    workflow.TaskObject['Describe Geometry'].Execute()

    # Updating boundaries
    workflow.TaskObject['Update Boundaries'].Execute()

    # Updating regions
    workflow.TaskObject['Update Regions'].Execute()

    # Add boundary layers
    workflow.TaskObject['Add Boundary Layers'].Arguments.set_state({r'LocalPrismPreferences': {r'Continuous': r'Stair Step',},})
    workflow.TaskObject['Add Boundary Layers'].AddChildAndUpdate()

    # Generate volume mesh
    workflow.TaskObject["Generate the Volume Mesh"].Arguments = {"VolumeFill": "poly-hexcore",}
    workflow.TaskObject['Generate the Volume Mesh'].Execute()

    # Switch to solution mode
    solver = meshing_session.switch_to_solver()

    # Assigning boundary conditions
    solver.setup.boundary_conditions.velocity_inlet['inlet'].vmag = 3.78
    # use this to find out available settings:
    # solver.setup.boundary_conditions.wall['moving-wall']()
    solver.setup.boundary_conditions.wall['moving-wall'].motion_bc = 'Moving Wall'
    solver.setup.boundary_conditions.wall['moving-wall'].vmag = 3.78
    solver.setup.boundary_conditions.wall['moving-wall'].wall_translation = [0, 0, 1]

    # Residuals
    solver.tui.solve.monitors.residual.plot("yes")

    # Hybrid initialization
    solver.solution.initialization.hybrid_initialize()

    # Set max iterations (may converge before this)
    solver.solution.run_calculation.iterate(iter_count = 50)

    # Set probe rake
    solver.tui.surface.rake_surface(
    "probe_rake",
    "0",        # x0
    "0",        # y0
    "0.0119",   # z0
    "0",        # x1
    "0.0015",   # y1
    "0.0119",   # z1
    "200",      # number of points
    )

    # Export rake data to csv
    solver.tui.file.export.ascii(
    "PyFluent_probe.csv",        # file name
    "probe_rake",           # surface name (can be multiple)
    "()",                   # leave surface zone menu
    "yes",                    # delimiter
    "velocity-magnitude",   # ASCII scalar (can be multiple)
    "()",                   # leave scalar menu
    "no",                   # node values (no), cell center values (yes)
    )

    # Exit solver
    solver.exit()

if __name__ == "__main__":
    tick = time.time()
    main()
    tock = time.time()
    print("Time taken: ", tock - tick)