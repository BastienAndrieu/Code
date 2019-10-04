ROOT = '/d/bandrieu/'#'/home/bastien/'

import bpy
import numpy

import sys
sys.path.append(ROOT + 'GitHub/Code/Python/')
import lib_blender_util as lbu
import lib_cadcheb as lcad

def add_primitive(xyz, name='primitive'):
    v, f = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
    obj = lbu.pydata_to_mesh(
        v,
        f,
        name=name
    )
    obj.show_wire = True
    obj.show_all_edges = True
    return obj

def random_rot():
    u = 2*numpy.random.rand(3)-1
    u = u/numpy.sqrt(u.dot(u))
    v = numpy.zeros(3)
    v[numpy.argmin(numpy.absolute(u))] = 1
    v = v - v.dot(u)
    v = v/numpy.sqrt(v.dot(v))
    w = numpy.cross(u,v)
    return numpy.vstack((u,v,w))

lbu.clear_scene(True, True, True)



"""
# Rectangle
rect = add_primitive(
    lcad.rectangle_cgl(
        numpy.random.rand(3),
        2.,
        1.,
        random_rot(),
        5,
        3),
    'rectangle'
    )


# Cone
rect = add_primitive(
    lcad.cone_cgl(
        numpy.random.rand(3),
        1.,
        0.3,
        1.5,
        random_rot(),
        0.0,
        numpy.pi/3,
        16,
        3),
    'cone'
    )
"""

# Cylinder
rect = add_primitive(
    lcad.cylinder_cgl(
        numpy.random.rand(3),
        0.3,
        1.5,
        random_rot(),
        0.0,
        numpy.pi/3,
        16,
        3),
    'cylinder'
    )

# Torus
rect = add_primitive(
    lcad.torus_cgl(
        numpy.random.rand(3),
        1.2,
        0.3,
        random_rot(),
        0.0,
        numpy.pi/3,
        numpy.pi/6,
        numpy.pi/2,
        16,
        8),
    'torus'
    )
