import trimesh
import polyscope as ps
import os

os.chdir(r"D:\STL testbed")

def real_scale(filename, width, length, height):
    model = trimesh.load_mesh(filename, enable_post_processing=True, solid=True)
    # https://github.com/mikedh/trimesh/issues/365 - notes on how trimesh scales
    scaling_model = model.copy()
    scaling_model.apply_scale((width, length, height))
    scaling_model.export("scale_" + str(width) + "-" + str(length) + "-" + str(height) + ".stl")

def show_model(filename):
    ps.init()
    myobj = trimesh.load_mesh(filename, enable_post_processing=True, solid=True) # Import Object fomr stl

    #tf = translation_matrix([1, 2, 3])
    #myobj = trimesh.base.Trimesh.apply_translation(myobj, (1000, 0, 20))
    #myobj.apply_translation((1000, 0, 0))
    #myobj.apply_scale((2, 1, 1))

    #myobj = trimesh.transformations()
    vertices = myobj.vertices
    faces = myobj.faces
    # visualize!
    #multimesh = trimesh.Scene(myobj)
    #multimesh.show()
    #multimesh.add_geometry(myobj, geom_name='mesh2')
    #trimesh.viewer.windowed.Sceneviewer(multimesh)
    ps_mesh = ps.register_surface_mesh(filename, vertices, faces)
    ps.show()

def main():
    real_scale("Real.stl", 1, 1, 2)
    #show_model("scale.stl")

if __name__ == "__main__":
    main()