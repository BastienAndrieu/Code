import numpy
from numpy import *
from numpy.polynomial.chebyshev import chebval

import matplotlib.pyplot as plt

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_chebyshev as cheb


#####################################
class Curve:
    def __init__(self, x):
        self.x = x
        self.xt = cheb.diff(x)
        return

    def eval(self, t):
        return chebval(t, self.x)

    def evald(self, t):
        return chebval(t, self.xt)

    def reparameterize(self, t0, t1):
        self.x = cheb.chgvar1(self.x, t0, t1)
        self.xt = cheb.diff(self.x)
        return 
#####################################
def radius_function(x, y):
    """
    return r, dr_dx, dr_dy
    """
    a = 0.022
    b = 0.4
    c = 12
    d = 0.3
    e = 5
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
        if abs(f) < (tol*r)**2:
            converged = True
            break

        df_dt = 2.0*(dot(do_dt, res) - dr_dt*(r + rprev))

        dt = -f/df_dt
        t += dt
    return converged, t, o, do_dt, r

########################################################
# DEFINE 
N = 8

c = zeros((N,2))
c[1,0] = 1
c[:,0] = c[:,0] + 0.4*(2*numpy.random.rand(N) - 1)
c[:,1] = (2*numpy.random.rand(N) - 1)
d = 1.0/arange(1,N+1)**3
for j in range(2):
    c[:,j] = c[:,j]*d

path = Curve(c)
########################################################


# PLOT CURVE WITH SAMPLE CIRCLES
"""
u = linspace(-1,1,200)
xy = path.eval(u)

xyc = path.eval(linspace(-1,1,20))

fig, ax = plt.subplots()

ax.plot(xy[0], xy[1], 'k')

ax.plot(xyc[0], xyc[1], 'r.')

for i in range(xyc.shape[1]):
    r, dr_dx, dr_dy = radius_function(xyc[0][i], xyc[1][i])
    ax.add_artist(
        plt.Circle(xyc[:,i], r, ec='g', fill=False)
    )

ax.set_aspect('equal')
plt.show()
"""
########################################################



########################################################
# PACK CIRCLES
TOL = 1e-2
FRAC_DT_PREDICTOR = 1
START_FROM_ENDPOINT = True
ADJUST_CURVE_EXTENT = False

centers_t = []
pend = path.eval(1)

# find 1st center
t = -1

# predictor
p = path.eval(t)
r, dr_dx, dr_dy = radius_function(p[0], p[1])
if r < 0: exit('r < 0')
dp_dt = path.evald(t)

if START_FROM_ENDPOINT:
    o = p
    do_dt = dp_dt
else:
    dt = FRAC_DT_PREDICTOR*r/norm2(dp_dt)
    print 'predictor dt = ', dt

    # Newton
    t += dt
    converged, t, o, do_dt, r = newton_circle_packing(
        path,
        t,
        p,
        0,
        TOL
    )

    if not converged:
        print 't = ', t
        print 'sqrt(|f|) = ', sqrt(abs(f)), ', TOL*r = ', TOL*r
        exit('first o not converged')

centers_t.append(t)

while t < 1:
    rprev = r
    p = o
    if sum((p - pend)**2) < rprev**2: break
    
    ## 1: find next center
    # predictor
    dt = FRAC_DT_PREDICTOR*(2*rprev)/norm2(do_dt)
    print 't = ', t, ', predictor dt = ', dt

    # Newton
    t += dt
    converged, t, o, do_dt, r = newton_circle_packing(
        path,
        t,
        p,
        rprev,
        TOL
    )
    
    if not converged:
        print 't = ', t
        print 'sqrt(|f|) = ', sqrt(abs(f)), ', TOL*r = ', TOL*r
        print 'o not converged'
        break
    else:
        centers_t.append(t)
########################################################





########################################################
# ADJUST CURVE EXTENT
if ADJUST_CURVE_EXTENT:
    tmax = centers_t[-1]
    centers_t = 2*(asarray(centers_t) + 1)/(tmax + 1) - 1
    path.reparameterize(-1, tmax)
########################################################







########################################################
# ENVELOPE OF CIRCLES
npts = 400
t = linspace(-1,1,npts)

xy = path.eval(t)
dxy_dt = path.evald(t)
sqr_norm_dxy_dt = sum(dxy_dt**2, axis=0)
norm_dxy_dt = sqrt(sqr_norm_dxy_dt)
inv_norm_dxy_dt = 1/norm_dxy_dt
normal = vstack([-dxy_dt[1], dxy_dt[0]])

r, dr_dx, dr_dy = radius_function(xy[0], xy[1])
dr_dt = dr_dx*dxy_dt[0] + dr_dy*dxy_dt[1]

eoc = []
q = zeros((2,npts))
for i in range(2):
    qn = sqrt(1 - dr_dt**2)*(-1)**i
    for j in range(2):
        q[j] = (qn*normal[j] - dr_dt*dxy_dt[j])*r*inv_norm_dxy_dt
    eoc.append(xy + q)
########################################################




########################################################
# VISUALIZE
centers_xy = path.eval(centers_t)

u = linspace(-1,1,200)
xy = path.eval(u)


fig, ax = plt.subplots()

ax.plot(xy[0], xy[1], 'k')

ax.plot(centers_xy[0], centers_xy[1], 'g.')

for i in range(centers_xy.shape[1]):
    r, dr_dx, dr_dy = radius_function(centers_xy[0][i], centers_xy[1][i])
    ax.add_artist(
        plt.Circle(
            centers_xy[:,i],
            r,
            ec='g',
            fill=False)
    )

for i in range(2):
    ax.plot(eoc[i][0], eoc[i][1], 'r')

ax.set_aspect('equal')
plt.show()
########################################################
