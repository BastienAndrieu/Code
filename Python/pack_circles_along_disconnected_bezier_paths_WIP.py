# -*- coding: utf-8 -*-

import numpy
from numpy import *

import matplotlib.pyplot as plt

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_bezier as lbez

#####################################
def norm2(u):
    return numpy.hypot(u[0], u[1])
#####################################
class Curve:
    def __init__(self, x):
        self.x = x
        self.update()
        return

    def eval(self, t):
        return lbez.eval_bezier_curve(self.x, t)

    def evald(self, t):
        return lbez.eval_bezier_curve(self.xt, t)
    
    def evald2(self, t):
        return lbez.eval_bezier_curve(self.xtt, t)

    def curvature(self, t):
        xt = self.evald(t)
        xtt = self.evald2(t)
        sqr_xt = sum(xt**2, axis=1)
        denom = maximum(EPSfp, minimum(HUGEfp, sqr_xt*sqrt(sqr_xt)))
        det = xt[:,0]*xtt[:,1] - xt[:,1]*xtt[:,0]
        return det/denom

    def update(self):
        self.xt = lbez.diff(self.x)
        self.xtt = lbez.diff(self.xt)
        return
#####################################
class Node:
    def __init__(self, co, ingoing=None, outgoing=None):
        self.co = numpy.asarray(co)
        if ingoing is None: ingoing = []
        self.ingoing = []
        if outgoing is None: outgoing = []
        self.outgoing = []
        return
#####################################
class Arc:
    def __init__(self, curve, verts=[None, None]):
        self.curve = curve
        self.verts = verts
        return
#####################################

########################################################
# READ INKSCAPE EXPORT
segments = []
fin = open('/d/bandrieu/Téléchargements/teapot_simple_solid_bcp.dat')
while True:
    line = fin.readline()
    if ("" == line): break # end of file
    nbp = int(line)
    
    bp = zeros((nbp,2))
    for i in range(nbp):
        bp[i] = [float(x) for x in fin.readline().split()]
    segments.append(Curve(bp))
fin.close()
########################################################

########################################################
# ADJUST (FLIP Y, SCALE DOWN, FIX SINGULAR ENDPOINTS)
scale = 1e-2
SPT = 1e-4
for curve in segments:
    curve.x[:,1] = -curve.x[:,1]
    curve.x = scale*curve.x
    if norm2(curve.x[0] - curve.x[1]) < SPT: curve.x = curve.x[1:]
    if norm2(curve.x[-1] - curve.x[-2]) < SPT: curve.x = curve.x[:-1]
    curve.update()
########################################################

########################################################
# CONCATENATE (GRAPH)
nodes = []
arcs = []
fig, ax = plt.subplots()
for curve in segments:
    arcs.append(Arc(curve))
    for jend, iend in enumerate([0,-1]):
        alias = None
        for nod in nodes:
            if norm2(curve.x[iend] - nod.co) < SPT:
                alias = nod
                break
        if alias is None:
            nodes.append(Node(co=curve.x[iend]))
            alias = nodes[-1]
        arcs[-1].verts[jend] = alias
        if jend == 0:
            alias.outgoing.append(arcs[-1])
        else:
            alias.ingoing.append(arcs[-1])
    print numpy.asarray([arcs[-1].verts[i].co for i in range(2)])
    xy = numpy.asarray([arcs[-1].verts[i].co for i in range(2)])
    ax.plot(xy[:,0], xy[:,1], 'k')

ax.set_aspect('equal')
plt.show()
########################################################

print '\n\n\n'

########################################################
fig, ax = plt.subplots()

for arc in arcs:
    xy = numpy.asarray([arc.verts[i].co for i in range(2)])
    print xy
    a = xy[0]
    v = xy[1] - a
    #ax.arrow(a[0], a[1], v[0], v[1], color='k')#, length_icludes_head=True)
    ax.plot(xy[:,0], xy[:,1], 'k')
    
for nod in nodes:
    ax.plot(nod.co[0], nod.co[1], 'go')
    for arc in nod.outgoing:
        xy = 0.25*(3*nod.co + arc.verts[1].co)
        ax.plot([nod.co[0], xy[0]], [nod.co[1], xy[1]], 'r')
    for arc in nod.ingoing:
        xy = 0.25*(3*nod.co + arc.verts[0].co)
        ax.plot([nod.co[0], xy[0]], [nod.co[1], xy[1]], 'b')

ax.set_aspect('equal')
plt.show()
########################################################
