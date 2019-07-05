# -*- coding: utf-8 -*-


######################################################
# TO DO:
# * REMOVE SELF-INTERSECTIONS OF THE ENVELOPE (--> lib_polyline)
# * FILL HOLES IN THE ENVELOPE WITH CIRCLE ARCS
######################################################

import numpy
from numpy import *

import matplotlib.pyplot as plt

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_bezier as lbez


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

    def update(self):
        self.xt = lbez.diff(self.x)
        return
#####################################
def radius_function(x, y):
    """
    return r, dr_dx, dr_dy
    """
    a = 0.28
    b = 0.2
    c = 2
    d = 0.1
    e = 1
    r = a*(1 + b*cos(c*x) + d*sin(e*y))
    dr_dx = -a*b*c*sin(c*x)
    dr_dy = a*d*e*cos(e*x)
    return r, dr_dx, dr_dy
#####################################
def norm2(u):
    return sqrt(sum(u**2))
#####################################
def newton_circle_packing(path, t, oprev, rprev, tol, itmax=100):
    converged = False
    for it in range(itmax):
        o = path.eval(t)
        do_dt = path.evald(t)
        r, dr_dx, dr_dy = radius_function(o[0], o[1])
        if r < 0: exit('newton_circle_packing: r < 0')
        dr_dt = do_dt[0]*dr_dx + do_dt[1]*dr_dy

        res = o - oprev

        f = dot(res, res) - (r + rprev)**2
        print '\t\tit.#%d, resf = %s' % (it, sqrt(abs(f))/r)
        if abs(f) < (tol*r)**2:
            converged = True
            break

        df_dt = 2.0*(dot(do_dt, res) - dr_dt*(r + rprev))

        dt = -f/df_dt
        #print '\t\t        dt = %s' % dt
        t += dt
    return converged, t, o, do_dt, r
#####################################
def diff_angle(u, v):
    return arctan2(u[1]*v[0] - u[0]*v[1], u[0]*v[0] + u[1]*v[1])
#####################################
def circle_arc_between_two_points(center, xy0, xy1, tolchord=1e-2):
    p0 = asarray(xy0) - asarray(center)
    p1 = asarray(xy1) - asarray(center)
    angle = arccos(min(1.0, max(-1.0, dot(p0,p1)/sqrt(dot(p0,p0)*dot(p1,p1)))))

    npts = max(2, int(0.5*angle/sqrt(tolchord*(2-tolchord))))
    
    t = linspace(0,1,npts)
    p = (outer(sin((1 - t)*angle), p0) + outer(sin(t*angle), p1))/sin(angle)
    for i in range(npts):
        p[i] = p[i] + center
    return p
#####################################

#####################################


########################################################
# DEFINE PATH
"""
path = [
    Curve(
        numpy.array([(5,3), (4,3), (3,4), (3,5)])
    ),
    Curve(
        numpy.array([(3,5), (2,5), (0,3), (0,2)])
    ),
    Curve(
        numpy.array([(0, 2), (1,0), (4,0), (5,3)])
    ),
]
"""
path = []
fin = open('/d/bandrieu/Téléchargements/bird_bcp.dat')
while True:
    line = fin.readline()
    if ("" == line): break # end of file
    nbp = int(line)
    
    bp = zeros((nbp,2))
    for i in range(nbp):
        bp[i] = [float(x) for x in fin.readline().split()]
    path.append(Curve(bp))
fin.close()
########################################################


########################################################
# ADJUST (FLIP Y, SCALE DOWN, FIX SINGULAR ENDPOINTS)
scale = 1e-2
SPT = 1e-4
for curve in path:
    curve.x[:,1] = -curve.x[:,1]
    curve.x = scale*curve.x
    if norm2(curve.x[0] - curve.x[1]) < SPT: curve.x = curve.x[1:]
    if norm2(curve.x[-1] - curve.x[-2]) < SPT: curve.x = curve.x[:-1]
    curve.update()
    print curve.x
########################################################


########################################################
"""
# PLOT CURVE WITH SAMPLE CIRCLES
t = linspace(0,1,100)
tc = linspace(0,1,20)


fig, ax = plt.subplots()

for curve in path:
    xy = curve.eval(t)
    xyc = curve.eval(tc)
    ax.plot(xy[:,0], xy[:,1], 'k')
    ax.plot(xyc[:,0], xyc[:,1], 'r.')

    for i in range(len(tc)):
        r, dr_dx, dr_dy = radius_function(xyc[i][0], xyc[i][1])
        ax.add_artist(
            plt.Circle(
                xyc[i],
                r,
                ec='g',
                fill=False
            )
        )

ax.set_aspect('equal')
plt.show()
"""
########################################################


########################################################
# PACK CIRCLES
TOL = 1e-1
FRAC_DT_PREDICTOR = 1
START_FROM_ENDPOINT = True

centers = []
pend = path[-1].eval(1.0)
rend, dr_dx, dr_dy = radius_function(pend[0], pend[1])

# find 1st center
t = 0.0
ipath = 0

# predictor
p = path[ipath].eval(t)
r, dr_dx, dr_dy = radius_function(p[0], p[1])
if r < 0: exit('r < 0')
dp_dt = path[ipath].evald(t)

if START_FROM_ENDPOINT:
    o = p
    do_dt = dp_dt
else:
    dt = FRAC_DT_PREDICTOR*r/norm2(dp_dt)
    print 'predictor dt = %s' % dt

    # Newton
    t += dt
    converged, t, o, do_dt, r = newton_circle_packing(
        path[ipath],
        t,
        p,
        0,
        TOL
    )
    if not converged:
        print 't = %s' % t
        exit('first o not converged')

centers.append([ipath, t])
dt = FRAC_DT_PREDICTOR*(2*r)/norm2(do_dt)

while True:
    print 'curve #%d/%d' % (ipath+1, len(path))
    while True:
        print 't = %s, predictor dt = %s' % (t, dt)

        # Newton
        told = t
        t += dt
        print '     predictor t = %s' % t
        converged, t, o, do_dt, r = newton_circle_packing(
            path[ipath],
            t,
            o,
            r,
            TOL
        )
        print '     converged t = %s    (dt = %s)' % (t, t - told)

        if not converged:
            print 't = %s' % t
            print 'o not converged'
            break
        else:
            if t > 1:
                break # move on to the next subpath
            else:
                centers.append([ipath, t])
                dt = FRAC_DT_PREDICTOR*(2*r)/norm2(do_dt)
    ipath += 1
    if ipath == len(path): break
    o = path[centers[-1][0]].eval(centers[-1][1])
    r, dr_dx, dr_dy = radius_function(o[0], o[1])
    t = 0.0
    q = path[ipath].eval(t)
    dq_dt = path[ipath].evald(t)
    # find point on linearized subpath at distance 2*r from last circle's center
    A = dot(dq_dt, dq_dt)
    B = dot(dq_dt, q-o)
    C = dot(q-o, q-o) - 4*r**2
    dt = (sqrt(B**2 - A*C) - B)/A
    s = q + dt*dq_dt
    print '|s - p| = %s, 2*r(p) = %s' % (norm2(s - o), 2*r)
    dt *= FRAC_DT_PREDICTOR
########################################################








########################################################
# ENVELOPE OF CIRCLES
npts = 400
t = linspace(0,1,npts)

eoc_branches = [[], []]
for curve in path:
    xy = curve.eval(t)
    dxy_dt = curve.evald(t)
    sqr_norm_dxy_dt = sum(dxy_dt**2, axis=1)
    inv_sqr_norm_dxy_dt = 1/sqr_norm_dxy_dt
    normal = vstack([dxy_dt[:,1], -dxy_dt[:,0]]).T
    
    r, dr_dx, dr_dy = radius_function(xy[:,0], xy[:,1])
    dr_dt = dr_dx*dxy_dt[:,0] + dr_dy*dxy_dt[:,1]

    q = zeros((npts,2))
    for i in range(2):
        qn = sqrt(sqr_norm_dxy_dt - dr_dt**2)*(-1)**i
        for j in range(2):
            q[:,j] = (qn*normal[:,j] - dr_dt*dxy_dt[:,j])*r*inv_sqr_norm_dxy_dt
        eoc_branches[i].append(xy + q)
########################################################


########################################################
# FILL HOLES IN ENVELOPE OF CIRCLES WITH ARCS
ncurves = len(path)
eoc = [empty((0,2), float), empty((0,2), float)]
for i in range(ncurves):
    for k in range(2):
        eoc[k] = vstack([eoc[k], eoc_branches[k][i][:-1]])
    j = (i+1)%ncurves
    xyi = path[i].eval(1.0)
    xyj = path[j].eval(0.0)
    if norm2(xyi - xyj) < SPT:
        for k in range(2):
            if norm2(eoc_branches[k][i][-1] - eoc_branches[k][j][0]) > SPT:
                # fill with circle arc
                ctr = 0.5*(xyi + xyj)
                xy_arc = circle_arc_between_two_points(
                    center=0.5*(xyi + xyj),
                    xy0=eoc_branches[k][i][-1],
                    xy1=eoc_branches[k][j][0],
                    tolchord=1e-3
                )
                eoc[k] = vstack([eoc[k], xy_arc[:-1]])
########################################################


########################################################
"""
# REMOVE SELF-INTERSECTION IN ENVELOPE OF CIRCLES -> ENVELOPE OF DISKS
eod = [empty((0,2), float), empty((0,2), float)]
inter = []
for k in range(2):
    n = len(eoc[k])
    splits = []
    for i in range(n-1):
        tngi = eoc[k][i+1] - eoc[k][i]
        nori = array([-tngi[1], tngi[0]])
        dii = dot(eoc[k][i], nori)
        for j in range(i+2,n-1):
            dji = dot(eoc[k][j], nori) - dii
            djp1i = dot(eoc[k][j+1], nori) - dii
            if dji*djp1i <= 0:
                tngj = eoc[k][j+1] - eoc[k][j]
                norj = array([-tngj[1], tngj[0]])
                djj = dot(eoc[k][j], norj)
                dij = dot(eoc[k][i], norj) - djj
                dip1j = dot(eoc[k][i+1], norj) - djj
                if dij*dip1j <= 0:
                    det = nori[0]*norj[1] - norj[0]*nori[1]
                    if abs(det) > 1e-7: 
                        invdet = 1/det
                        x = (dii*norj[1] - djj*nori[1])*invdet
                        y = (nori[0]*djj - norj[0]*dii)*invdet
                        splits.append((i, j, x, y))
                        inter.append((x,y))
    if len(splits) < 1:
        eod[k] = eoc[k]
    else:
        eod[k] = eoc[k][:splits[0][0]+1]
        for isplit in range(len(splits)-1):
            i, j, x, y = splits[isplit]
            eod[k] = vstack([eod[k], [x,y]])
            eod[k] = vstack([eod[k], eoc[k][j+1:splits[isplit+1][0]+1]])
        eod[k] = vstack([eod[k], eoc[k][splits[-1][1]+1:]])
"""
########################################################

########################################################
# VISUALIZE
print centers
centers_xy = numpy.array([path[ipath].eval(t) for ipath, t in centers])


t = linspace(0,1,100)

fig, ax = plt.subplots()

for curve in path:
    xy = curve.eval(t)
    ax.plot(xy[:,0], xy[:,1], 'k')

if True:
    for i in range(len(centers_xy)):
        r, dr_dx, dr_dy = radius_function(centers_xy[i][0], centers_xy[i][1])
        ax.add_artist(
            plt.Circle(
                centers_xy[i],
                r,
                ec='g',
                fill=False
            )
        )

#ax.plot(centers_xy[:,0], centers_xy[:,1], 'b.')
if True:
    cl = ['r', 'b']
    for i in range(2):
        """
        for xy in eoc[i]:
            ax.plot(xy[:,0], xy[:,1], color=cl[i])
        """
        ax.plot(eoc[i][:,0], eoc[i][:,1], color=cl[i])

#for x, y in inter:
#    ax.plot(x, y, 'k*')
    
ax.set_aspect('equal')
plt.show()
########################################################
