# -*- coding: utf-8 -*-

import numpy
from numpy import *

import matplotlib.pyplot as plt

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_bezier as lbez
#from lib_compgeom import angle_between_vectors_2d


EPSfp = 1e-15
HUGEfp = 1e15

EPSalpha = 1e-6
EPSspt = 1e-4



#####################################
def radius_function(x, y):
    """
    return r, dr_dx, dr_dy
    """
    a = 0.08#0.28#
    b = 0.2
    c = 2
    d = 0.13
    e = 1.5
    ox = 0.1
    oy = 0.2
    r = a*(1 + b*sin(c*x + ox) + d*cos(e*y + oy))
    dr_dx = a*b*c*cos(c*x + ox)
    dr_dy = -a*d*e*sin(e*y + oy)
    d2r_dx2 = -a*b*c*c*sin(c*x + ox)
    d2r_dy2 = -a*d*e*e*cos(e*y + oy)
    d2r_dxdy = 0*x
    return r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy
#####################################









#####################################
def norm2(u):
    return numpy.hypot(u[0], u[1])
#####################################
def perp(u):
    return numpy.array([-u[1], u[0]])
#####################################
def angle_between_vectors_2d(u, v):
    return numpy.arccos(
        min(1.0,
            max(-1.0,
                u.dot(v) / numpy.sqrt(u.dot(u)*v.dot(v))
            )
        )
    )
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

    def curvature(self, t, xt=None, xtt=None):
        if xt is None: xt = self.evald(t)
        if xtt is None: xtt = self.evald2(t)
        sqr_xt = sum(xt**2, axis=1)
        denom = maximum(EPSfp, minimum(HUGEfp, sqr_xt*sqrt(sqr_xt)))
        det = xt[:,0]*xtt[:,1] - xt[:,1]*xtt[:,0]
        return det/denom

    def update(self):
        self.xt = lbez.diff(self.x)
        self.xtt = lbez.diff(self.xt)
        return

    def make_envelope_propre(self, t):
        xy = self.eval(t)
        dxy_dt = self.evald(t)
        sqr_norm_dxy_dt = sum(dxy_dt**2, axis=1)
        inv_sqr_norm_dxy_dt = 1/sqr_norm_dxy_dt
        normal = vstack([dxy_dt[:,1], -dxy_dt[:,0]]).T

        r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(xy[:,0], xy[:,1])
        dr_dt = dr_dx*dxy_dt[:,0] + dr_dy*dxy_dt[:,1]

        n = len(t)
        a = curve.alpha(t, dxy_dt, r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy)
        
        self.eoc = [numpy.zeros((n,2)), numpy.zeros((n,2))]
        self.eod = [[], []]
        for iside in range(2):
            qn = sqrt(sqr_norm_dxy_dt - dr_dt**2)*(-1)**iside
            for j in range(2):
                self.eoc[iside][:,j] = (qn*normal[:,j] - dr_dt*dxy_dt[:,j])*r*inv_sqr_norm_dxy_dt
            self.eoc[iside] = self.eoc[iside] + xy
            # look for local self-intersections (folding)
            if numpy.amin(a[iside]) > EPSalpha:
                self.eod[iside] = [self.eoc[iside].copy()]
            else:
                flipped = []
                ihead = 0
                while ihead < n:
                    #print 'ihead = %d/%d' % (ihead,n)
                    if a[iside][ihead] > EPSalpha:
                        ihead += 1
                        continue
                    for itail in range(ihead,n-1):
                        if a[iside][itail+1] > EPSalpha:
                            break
                    flipped.append([ihead,itail])
                    ihead = itail + 1
                    if ihead == n-1: break
                #
                #print '\t\t\tside ', iside, ', flipped = ', flipped
                if flipped[0][0] > 0:
                    self.eod[iside].append(self.eoc[iside][:flipped[0][0]])
                for j in range(len(flipped)-1):
                    self.eod[iside].append(self.eoc[iside][flipped[j][1]+1:flipped[j+1][0]])
                self.eod[iside].append(self.eoc[iside][flipped[-1][1]+1:n])
        return
    #####################
    def alpha(self, t, dxy_dt=None, r=None, dr_dx=None, dr_dy=None, d2r_dx2=None, d2r_dy2=None, d2r_dxdy=None):
        # position and derivatives
        if dxy_dt is None: dxy_dt = self.evald(t)
        d2xy_dt2 = self.evald2(t)

        # curvature
        k = self.curvature(t, dxy_dt, d2xy_dt2)

        # radius function and derivatives
        if r is None or dr_dx is None or dr_dy is None or d2r_dx2 is None or d2r_dy2 is None or d2r_dxdy is None:
            xy = self.eval(t)
            r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(xy[:,0], xy[:,1])
        dr_dt = dr_dx*dxy_dt[:,0] + dr_dy*dxy_dt[:,1]
        d2r_dt2 = dr_dx*d2xy_dt2[:,0] + dr_dy*d2xy_dt2[:,1] + (d2r_dx2 + d2r_dxdy)*dxy_dt[:,0] + (d2r_dy2 + d2r_dxdy)*dxy_dt[:,1]
        
        sqrtterm = numpy.sqrt(1 - dr_dt**2)
        commonterm = sqrtterm - dr_dt*d2r_dt2/sqrtterm
        return [commonterm + k*r, commonterm - k*r]
#####################################
def intersect_polylines(curves):
    n0 = len(curves[0])
    n1 = len(curves[1])
    intersections = []
    for i in range(n0-1,0,-1):
        tngi = curves[0][i] - curves[0][i-1]
        nori = perp(tngi)
        dii = nori.dot(curves[0][i])
        for j in range(n1-1):
            dji = nori.dot(curves[1][j]) - dii
            djp1i = nori.dot(curves[1][j+1]) - dii
            if dji*djp1i < 0:
                tngj = curves[1][j+1] - curves[1][j]
                norj = perp(tngj)
                djj = norj.dot(curves[1][j])
                dij = norj.dot(curves[0][i]) - djj
                dim1j = norj.dot(curves[0][i-1]) - djj
                if dij*dim1j < 0:
                    det = nori[0]*norj[1] - norj[0]*nori[1]
                    if abs(det) > 1e-7:
                        invdet = 1/det
                        x = (dii*norj[1] - djj*nori[1])*invdet
                        y = (nori[0]*djj - norj[0]*dii)*invdet
                        intersections.append([i, j, x, y])
    return intersections
#####################################
def read_connected_path(filename):
    segments = []
    f = open(filename, 'r')
    n = int(f.readline())
    for i in range(n):
        ni = int(f.readline())
        bp = zeros((ni,2))
        for j in range(ni):
            bp[j] = [float(x) for x in f.readline().split()]
        segments.append(Curve(bp))
    f.close()
    return segments
#####################################


def newton_circle_packing(path, t, oprev, rprev, tol, itmax=20):
    converged = False
    for it in range(itmax):
        o = path.eval(t)
        do_dt = path.evald(t)
        r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(o[0], o[1])
        if r < 0: exit('newton_circle_packing: r < 0')
        dr_dt = do_dt[0]*dr_dx + do_dt[1]*dr_dy

        res = o - oprev

        f = dot(res, res) - (r + rprev + SPACE_BETWEEN_CIRCLES)**2
        #print '\t\tit.#%d, resf = %s' % (it, sqrt(abs(f))/r)
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
    p0 = xy0 - center
    p1 = xy1 - center
    angle = angle_between_vectors_2d(p0, p1)%(2*numpy.pi)

    npts = max(2, int(0.5*abs(angle)/sqrt(tolchord*(2-tolchord))))
    
    t = linspace(0,1,npts)
    s = sin(angle)
    if abs(s) < 1e-15:
        #print 'forcing linear interpolation'
        p = outer(1 - t, p0) + outer(t, p1)
    else:
        # slerp
        p = (outer(sin((1 - t)*angle), p0) + outer(sin(t*angle), p1))/s
    for i in range(npts):
        p[i] = p[i] + center
    return p
#####################################




########################################################
# READ INKSCAPE EXPORT
if True:
    elems = []
    #for name in ['body', 'spout', 'handle_out', 'handle_in']:
    #for name in ['body', 'spout', 'handle']:
    for name in ['outer', 'inner']:
        elems.append(read_connected_path('/d/bandrieu/Téléchargements/teapot_simple_'+name+'_bcp.dat'))
else:
    elems = [read_connected_path('/d/bandrieu/Téléchargements/teapot_simple_outer_bcp.dat')]

########################################################
# PLOT CURRENT INTERFACE
t = linspace(0,1,100)
tc = linspace(0,1,5)

if False:
    fig, ax = plt.subplots()

    for ipath, path in enumerate(elems):
        for icurve, curve in enumerate(path):#outer[:10] + inner:
            xy = curve.eval(t)
            xyc = curve.eval(tc)
            ax.plot(xy[:,0], xy[:,1], 'k')

            for i in range(len(tc)):
                r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(xyc[i][0], xyc[i][1])
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
    exit()
########################################################



########################################################
# PACK CIRCLES
TOL = 1e-1
FRAC_DT_PREDICTOR = 1
START_FROM_ENDPOINT = True
SPACE_BETWEEN_CIRCLES = 0#1e-3#

centers = []
centers_xy = []

"""
for numpath, path in enumerate(elems):#,inner]):
    print 'path #%d' % (numpath+1)
    pend = path[-1].eval(1.0)
    rend, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(pend[0], pend[1])

    # find 1st center
    t = 0.0
    ipath = 0

    # predictor
    p = path[ipath].eval(t)
    r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(p[0], p[1])
    if r < 0: exit('   r < 0')
    dp_dt = path[ipath].evald(t)

    if START_FROM_ENDPOINT:
        o = p
        do_dt = dp_dt
    else:
        dt = FRAC_DT_PREDICTOR*r/norm2(dp_dt)
        #print '   predictor dt = %s' % dt

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
            #print '   t = %s' % t
            exit('   first o not converged')

    centers.append([ipath, t])
    dt = FRAC_DT_PREDICTOR*(2*r)/norm2(do_dt)

    keep_running = True
    while keep_running:
        print '   curve #%d/%d' % (ipath+1, len(path))
        while True:
            print '   t = %s, predictor dt = %s' % (t, dt)

            # Newton
            told = t
            t += dt
            print '     predictor t = %s' % t
            if t > 2:
                print '     too large --> move on to next segment'
                break
            elif t < 0:
                t = 0
                print '     force t = 0'
            converged, t, o, do_dt, r = newton_circle_packing(
                path[ipath],
                t,
                o,
                r,
                TOL
            )
            #print '     converged t = %s    (dt = %s)' % (t, t - told)

            if not converged:
                print '   t = %s' % t
                print '   o not converged'
                keep_running = False
                break
            else:
                if t > 1:
                    break # move on to the next subpath
                else:
                    centers.append([ipath, t])
                    centers_xy.append(path[ipath].eval(t))
                    dt = FRAC_DT_PREDICTOR*(2*r)/norm2(do_dt)
        ipath += 1
        if ipath == len(path): break
        o = path[centers[-1][0]].eval(centers[-1][1])
        r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(o[0], o[1])
        t = 0.0
        q = path[ipath].eval(t)
        dq_dt = path[ipath].evald(t)
        # find point on linearized subpath at distance 2*r from last circle's center
        A = dot(dq_dt, dq_dt)
        B = dot(dq_dt, q-o)
        C = dot(q-o, q-o) - 4*r**2
        dt = (sqrt(max(0, B**2 - A*C)) - B)/A
        s = q + dt*dq_dt
        #print '|s - p| = %s, 2*r(p) = %s' % (norm2(s - o), 2*r)
        dt *= FRAC_DT_PREDICTOR
"""
########################################################


########################################################
# ENVELOPPES (PROPRES) DES CERCLES
npts = 100
t = linspace(0,1,npts)

eoc_paths = []
for ipath, path in enumerate(elems):
    print 'ipath = %d' % ipath
    eoc_branches = [[], []]
    for icurve, curve in enumerate(path):
        print '\ticurve = %d' % icurve
        curve.make_envelope_propre(t)
########################################################


########################################################
# PREPARES ENVELOPES OF CIRCLES/DISKS
arcs = []
intersections = []
eod = [[[] for iside in range(2)] for path in elems]
for ipath, path in enumerate(elems):
    ncurves = len(path)
    for icurve, curvei in enumerate(path):
        for iside in range(2):
            eod[ipath][iside].append(curvei.eoc[iside])
        # look for self-intersections
        # ...
        # look for intersections at sharp joints
        jcurve = (icurve + 1)%ncurves
        curvej = path[jcurve]
        dxyi = curvei.evald(1.0)
        dxyj = curvej.evald(0.0)
        angle = numpy.degrees(angle_between_vectors_2d(dxyi, dxyj))
        if angle > 2.0: # sharp joint
            # MAKE ARCS
            arc = Curve(numpy.zeros((1,2)))
            arc.x[0] = curvei.x[-1]
            arc.eoc = []
            arc.eod = [[], []]
            for iside in range(2):
                xy_arc = circle_arc_between_two_points(
                    center=arc.x[0].copy(),
                    xy0=curvei.eoc[iside][-1],
                    xy1=curvej.eoc[iside][0],
                    tolchord=1e-4
                )
                arc.eoc.append(xy_arc)
            arcs.append([ipath, icurve, arc])
            # FIND INTERSECTION
            sgn = dxyi[0]*dxyj[1] - dxyi[1]*dxyj[0]
            if sgn < 0:
                inter_ij = intersect_polylines([c.eoc[0] for c in [curvei, curvej]])
                intersections.append([ipath, 0, icurve, jcurve, inter_ij])
                # add arc on side 1 to eod...
                eod[ipath][1].append(arc.eoc[1])
                arc.eod[1] = [arc.eoc[1].copy()]
            else:
                inter_ij = intersect_polylines([c.eoc[1] for c in [curvei, curvej]])
                intersections.append([ipath, 1, icurve, jcurve, inter_ij])
                # add arc on side 0 to eod...
                eod[ipath][0].append(arc.eoc[0])
                arc.eod[0] = [arc.eoc[0].copy()]

########################################################
# ENVELOPE OF CIRCLES
for i in range(len(arcs)):
    ipath, icurve, arc = arcs[i]
    elems[ipath].insert(icurve+1, arc)
    for j in range(i+1,len(arcs)):
        if arcs[j][0] == ipath:
            arcs[j][1] += 1
########################################################




"""
side = ['ext.', 'int.']
for ipath, iside, icurve, jcurve, points in intersections:
    print 'path#%d: curves #%d and #%d intersect (%s)' % (ipath, icurve, jcurve, side[iside])
    for point in points:
        print '\tat (%s, %s)' % (point[2], point[3])
"""
########################################################




"""
########################################################
# FILL HOLES IN ENVELOPE OF CIRCLES WITH ARCS
SPT = 1e-4

arcs = []
pairs = []
for ipath, path in enumerate(elems):
    ncurves = len(path)
    for icurve, curve in enumerate(path):
        jcurve = (icurve + 1)%ncurves
        dxyi = curve.evald(1.0)
        dxyj = path[jcurve].evald(0.0)
        sgn = dxyi[0]*dxyj[1] - dxyi[1]*dxyj[0]
        angle = numpy.degrees(angle_between_vectors_2d(dxyi, dxyj))
        if angle > 2.0:
            sgn = dxyi[0]*dxyj[1] - dxyi[1]*dxyj[0]
            if sgn > 0:
                pairs.append([ipath, 1, icurve, jcurve+1])
            else:
                pairs.append([ipath, 0, icurve, jcurve+1])
            for iside in range(2):
                xyei = eoc_paths[ipath][iside][icurve][-1]
                xyej = eoc_paths[ipath][iside][jcurve][0]
                xy_arc = circle_arc_between_two_points(
                    center=curve.x[-1],
                    xy0=xyei,
                    xy1=xyej,
                    tolchord=1e-4
                )
                arcs.append([ipath, iside, icurve, xy_arc])

for i in range(len(arcs)):
    ipath, iside, icurve, xy_arc = arcs[i]
    eoc_paths[ipath][iside].insert(icurve+1, xy_arc)
    for j in range(i+1,len(arcs)):
        if arcs[j][0] == ipath and arcs[j][1] == iside:
            arcs[j][2] += 1
########################################################


########################################################
# REMOVE SELF-INTERSECTION IN ENVELOPE OF CIRCLES -> ENVELOPE OF DISKS
# 1)
#pairs =[]
xy_inter = numpy.empty((0,2))
for ipath, iside, icurve, jcurve in pairs:
    curvei = eoc_paths[ipath][iside][icurve]
    curvej = eoc_paths[ipath][iside][jcurve]
    ni = len(curvei)
    nj = len(curvej)
    for i in range(ni-1,0,-1):
        tngi = curvei[i] - curvei[i-1]
        nori = perp(tngi)
        dii = curvei[i].dot(nori)
        for j in range(nj-1):
            dji = curvej[j].dot(nori) - dii
            djp1i = curvej[j+1].dot(nori) - dii
            if dji*djp1i <= 0:
                tngj = curvej[j+1]- curvej[j]
                norj = perp(tngj)
                djj = curvej[j].dot(norj)
                dij = curvei[i].dot(norj) - djj
                dim1j = curvei[i-1].dot(norj) - djj
                if dij*dim1j <= 0:
                    det = nori[0]*norj[1] - norj[0]*nori[1]
                    if abs(det) > 1e-7: 
                        invdet = 1/det
                        x = (dii*norj[1] - djj*nori[1])*invdet
                        y = (nori[0]*djj - norj[0]*dii)*invdet
                        xy_inter = numpy.vstack([xy_inter, [x,y]])
# 2)
########################################################
"""

########################################################
# REMOVE SELF-INTERSECTIONS
eod = []
for path in elems:
    eod_path = [[], []]
    for curve in path:
        for iside, eod_side in enumerate(curve.eod):
            for xy in eod_side:
                eod_path[iside].append(xy)
    eod.append(eod_path)


for ipath in range(len(eod)):
    for iside in range(2):
        icurve = 0
        while True:
            ncurves = len(eod[ipath][iside])
            if ncurves < 2: break
            #print '%d/%d' % (icurve, ncurves)
            curvei = eod[ipath][iside][icurve]
            jcurve = (icurve + 1)%ncurves
            curvej = eod[ipath][iside][jcurve]
            #
            if norm2(curvei[-1] - curvej[0]) > EPSspt:
                #print ipath, iside, icurve, norm2(curvei[-1] - curvej[0])
                if icurve+1 == len(eod[ipath][iside]):
                    break
                else:
                    icurve = jcurve
                    continue
            else:
                eod[ipath][iside][icurve] = numpy.vstack(
                    [
                        eod[ipath][iside][icurve][:],
                        eod[ipath][iside][jcurve][1:]
                    ]
                )
                del eod[ipath][iside][jcurve]
                if icurve == len(eod[ipath][iside]): break
                continue


xy_inter = []
for ipath in range(len(eod)):
    for iside in range(2):
        ncurves = len(eod[ipath][iside])
        for icurve in range(ncurves):
            curvei = eod[ipath][iside][icurve]
            jcurve = (icurve + 1)%ncurves
            curvej = eod[ipath][iside][jcurve]
            if norm2(curvei[-1] - curvej[0]) > EPSspt:
                intersections = intersect_polylines([curvei, curvej])
                for i, j, x, y in intersections:
                    xy_inter.append([x,y])
                if len(intersections) == 1:
                    i, j, x, y = intersections[0]
                    eod[ipath][iside][icurve] = curvei[:i+1]
                    eod[ipath][iside][jcurve] = curvej[j:]
                    eod[ipath][iside][icurve][-1] = [x,y]
                    eod[ipath][iside][jcurve][0] = [x,y]
                elif len(intersections) == 2:
                    ipoints = [intersections[k][0] for k in range(2)]
                    if ipoints[0] < ipoints[1]:
                        eod[ipath][iside][icurve] = curvei[ipoints[0]:ipoints[1]+1]
                        eod[ipath][iside][icurve][0] = intersections[0][2:4]
                        eod[ipath][iside][icurve][-1] = intersections[1][2:4]
                    else:
                        eod[ipath][iside][icurve] = curvei[ipoints[1]:ipoints[0]+1]
                        eod[ipath][iside][icurve][0] = intersections[1][2:4]
                        eod[ipath][iside][icurve][-1] = intersections[0][2:4]
                    #
                    jpoints = [intersections[k][1] for k in range(2)]
                    if jpoints[0] < jpoints[1]:
                        eod[ipath][iside][jcurve] = curvej[jpoints[0]:jpoints[1]+1]
                        eod[ipath][iside][jcurve][0] = intersections[0][2:4]
                        eod[ipath][iside][jcurve][-1] = intersections[1][2:4]
                    else:
                        eod[ipath][iside][jcurve] = curvej[jpoints[1]:jpoints[0]+1]
                        eod[ipath][iside][jcurve][0] = intersections[1][2:4]
                        eod[ipath][iside][jcurve][-1] = intersections[0][2:4]
########################################################



########################################################
# VISUALIZE
cl = ['r', 'b']
#centers_xy = numpy.array([path[ipath].eval(t) for ipath, t in centers])
centers_xy = numpy.asarray(centers_xy)

t = linspace(0,1,100)
im = int(len(t)/2)

"""
fig, ax = plt.subplots()
for ipath, path in enumerate(elems):
    for icurve, curve in enumerate(path):#outer + inner:
        xy = curve.eval(t)
        ax.plot(xy[:,0], xy[:,1], 'k')
        #
        critical = []
        a = curve.alpha(t)
        for iside in range(2):
            ia = numpy.where(a[iside] < EPSalpha)
            critical.append(ia)
            ax.plot(xy[ia,0], xy[ia,1], '.', color=cl[iside])
        #ax.text(xy[im,0], xy[im,1], '%d|%d' % (ipath, icurve), fontsize=8, color='g')
        #
        for i, xy in enumerate(curve.eoc):
            ax.plot(xy[:,0], xy[:,1], '-', color=cl[i])
            ax.plot(xy[critical[i],0], xy[critical[i],1], 'k.')
if True:
    for i in range(len(centers_xy)):
        r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(centers_xy[i][0], centers_xy[i][1])
        ax.add_artist(
            plt.Circle(
                centers_xy[i],
                r,
                ec='g',
                fc='g',
                alpha=0.5
            )
        )

if False:
    for path in eoc_paths:
        for i in range(2):
            for xy in path[i]:
                ax.plot(xy[:,0], xy[:,1], color=cl[i])

#ax.plot(xy_inter[:,0], xy_inter[:,1], 'go')
                
ax.set_aspect('equal')
plt.show()
"""
########################################################



fig, ax = plt.subplots()

for path in elems:
    for curve in path:#
        xy = curve.eval(t)
        ax.plot(xy[:,0], xy[:,1], 'k-')

if True:
    for i in range(len(centers_xy)):
        r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(centers_xy[i][0], centers_xy[i][1])
        ax.add_artist(
            plt.Circle(
                centers_xy[i],
                r,
                ec='g',
                fc='g',
                alpha=0.5
            )
        )
        
for path in eod:
    for iside in range(2):
        for xy in path[iside]:
            ax.plot(xy[:,0], xy[:,1], '-', color=cl[iside])

xy_inter = numpy.asarray(xy_inter)
#ax.plot(xy_inter[:,0], xy_inter[:,1], 'g.')


ax.set_aspect('equal')
plt.show()




########################################################
# TIKZ EXPORT
pthcode = ''
########################################################
