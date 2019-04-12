import numpy

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_halfedge as lhe

import matplotlib.pyplot as plt
#############################################

f2v = []
f = open('f2v.dat','r')
for l in f:
    f2v.append([int(a) for a in l.split()])
f.close()


nv = 0
for f in f2v:
    for v in f:
        nv = max(nv, v)
nv += 1


# Laplacian matrix
L = numpy.zeros((nv,nv))
D = numpy.zeros(nv)
uv = numpy.zeros((nv,2))

f = open('conformal_laplacian_matrix.dat','r')
for l in f:
    s = l.split()
    i = int(s[0])
    j = int(s[1])
    w = float(s[2])
    L[i,j] = L[i,j] + w
    L[j,i] = L[j,i] + w
f.close()
"""
for i in range(nv):
    D[i] = numpy.sum(L[i,:])
L = numpy.diag(D) - L
"""


"""
fig, ax = plt.subplots()
ax.spy(L)
plt.show()
"""

# boundary
boundary = []
f = open('boundary_vertices.dat','r')
for l in f:
    boundary.append(int(l))
f.close()

# fix boundary
nb = len(boundary)
t = numpy.linspace(0,2.0*numpy.pi,nb+1)
rb = numpy.ones(len(t))#1.0 + 0.1*numpy.cos(3.0*t)
ub = rb*numpy.cos(t)
vb = rb*numpy.sin(t)
for i, j in enumerate(boundary):
    #L[j,:] = 0.0
    #L[j,j] = 1.0
    uv[j,0] = ub[i]
    uv[j,1] = vb[i]

# solve
uv = numpy.linalg.solve(L, uv)


# make halfedge mesh for plot
mesh = lhe.pydata_to_SurfaceMesh(uv, f2v)

# plot
fig, ax = plt.subplots()
lhe.plot_mesh(mesh,
              ax,
              faces=False,
              halfedges=False,
              vertices=False,
              v2h=False,
              v2f=False,
              count_from_1=False)

ax.set_aspect('equal')
plt.show()


###########################################

import lib_sparse as lsp
f = open('conformal_laplacian_matrix.dat','r')
L = []
for l in f:
    s = l.split()
    i = int(s[0])
    j = int(s[1])
    w = float(s[2])
    L.append([i,j,w])
f.close()

# fix boundary
uv = lsp.init_vector(m=nv,n=2,val=0.0)
for i, j in enumerate(boundary):
    uv[j][0] = float(ub[i])
    uv[j][1] = float(vb[i])

uv = lsp.solve_jacobi(L, uv, tol=1e-7, itmax=100, eps=1e-7)

# make halfedge mesh for plot
mesh = lhe.pydata_to_SurfaceMesh(uv, f2v)

# plot
fig, ax = plt.subplots()
lhe.plot_mesh(mesh,
              ax,
              faces=False,
              halfedges=False,
              vertices=False,
              v2h=False,
              v2f=False,
              count_from_1=False)

ax.set_aspect('equal')
plt.show()
