import math
import stl
from stl import mesh
import numpy
import os
import time

os.chdir(r"D:\STL testbed")

# https://pypi.org/project/numpy-stl/#:~:text=Combining%20multiple%20STL%20files

# find the max dimensions, so we can know the bounding box, getting the height,
# width, length (because these are the step size)...
def find_mins_maxs(obj):
    minx = obj.x.min()
    maxx = obj.x.max()
    miny = obj.y.min()
    maxy = obj.y.max()
    minz = obj.z.min()
    maxz = obj.z.max()
    return minx, maxx, miny, maxy, minz, maxz


def translate(_solid, step, padding, multiplier, axis):
    if 'x' == axis:
        items = 0, 3, 6
    elif 'y' == axis:
        items = 1, 4, 7
    elif 'z' == axis:
        items = 2, 5, 8
    else:
        raise RuntimeError('Unknown axis %r, expected x, y or z' % axis)

    # _solid.points.shape == [:, ((x, y, z), (x, y, z), (x, y, z))]
    _solid.points[:, items] += (step * multiplier) + (padding * multiplier)


def copy_obj(obj, dims, num_rows, num_cols, num_layers):
    w, l, h = dims
    copies = []
    for layer in range(num_layers):
        for row in range(num_rows):
            for col in range(num_cols):
                # skip the position where original being copied is
                if row == 0 and col == 0 and layer == 0:
                    continue
                _copy = mesh.Mesh(obj.data.copy())
                # pad the space between objects by 10% of the dimension being
                # translated
                if col != 0:
                    translate(_copy, w, 0, col, 'x')     ##### MULTIPLIER ####### 
                if row != 0:
                    translate(_copy, l, 0, row, 'y')     ##### MULTIPLIER ####### 
                if layer != 0:
                    translate(_copy, h, 0, layer, 'z')   ##### MULTIPLIER ####### 
                copies.append(_copy)
    return copies

tick = time.time()
# Using an existing stl file:
main_body = mesh.Mesh.from_file('Real.stl')

# rotate along Y
main_body.rotate([0.0, 0.0, 0.0], math.radians(90))

minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(main_body)
w1 = maxx - minx
l1 = maxy - miny
h1 = maxz - minz
copies = copy_obj(main_body, (w1, l1, h1), 5, 45, 1)

# I wanted to add another related STL to the final STL
#twist_lock = mesh.Mesh.from_file('Real2.stl')
#minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(twist_lock)
#w2 = maxx - minx
#l2 = maxy - miny
#h2 = maxz - minz
#translate(twist_lock, w1, w1 / 10., 3, 'x')
#copies2 = copy_obj(twist_lock, (w2, l2, h2), 2, 2, 1)
#combined = mesh.Mesh(numpy.concatenate([main_body.data, twist_lock.data] +
#                                    [copy.data for copy in copies] +
#                                    [copy.data for copy in copies2]))

combined = mesh.Mesh(numpy.concatenate([main_body.data] +
                                    [copy.data for copy in copies]))

combined.save('combined binary 5-45.stl', mode=stl.Mode.BINARY)  # save as ASCII or BINARY

tock = time.time()
print(tock - tick)

#############################
## Possible NumPy solution
#############################
#a = np.array([0, 6, 7, 9, 1, 2, 4, 6, 2, 7])


#b = a.reshape(2, 5)
#final = np.repeat(b, 3, axis=0)


#test = np.tile(b, (3, 1))
#final = np.concatenate((test[::2], test[1::2]))