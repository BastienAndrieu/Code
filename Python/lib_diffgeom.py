import math
#############################################
def dot(u, v):
    udotv = 0.0
    for i in range(min(len(u), len(v))):
        udotv += u[i]*v[i]
    return udotv
#############################################
def cross(u, v):
    return [u[1]*v[2] - u[2]*v[1],
            u[2]*v[0] - u[0]*v[2],
            u[0]*v[1] - u[1]*v[0]]
#############################################
def scale(v, s):
    return [x*s for x in v]
#############################################
def first_fundamental_form(dxyz_du, dxyz_dv):
    return dot(dxyz_du, dxyz_du), dot(dxyz_du, dxyz_dv), dot(dxyz_dv, dxyz_dv)
#############################################
def second_fundamental_form(d2xyz_du2, d2xyz_dudv, d2xyz_dv2, n):
    return dot(d2xyz_du2, n), dot(d2xyz_dudv, n), dot(d2xyz_dv2, n)
#############################################
def det_symmetric_tensor(a11, a21, a22):
    return a11*a22 - a21**2
#############################################
def unit_normal(dxyz_du, dxyz_dv):
    E, F, G = first_fundamental_form(dxyz_du, dxyz_dv)
    return cross(dxyz_du, dxyz_dv)/math.sqrt(det_symmetric_tensor(E, F, G))
#############################################
def all_geomdiff(dxyz_du, dxyz_dv, d2xyz_du2, d2xyz_dudv, d2xyz_dv2):
    E, F, G = first_fundamental_form(dxyz_du, dxyz_dv)
    detEFG = det_symmetric_tensor(E, F, G)
    if detEFG < 1e-14:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    invdetEFG = 1.0/det_symmetric_tensor(E, F, G)
    n = scale(cross(dxyz_du, dxyz_dv), math.sqrt(invdetEFG))
    #
    L, M, N = second_fundamental_form(d2xyz_du2, d2xyz_dudv, d2xyz_dv2, n)
    #
    Kgauss = det_symmetric_tensor(L, M, N)*invdetEFG
    Kmean = 0.5*(E*N + G*L - 2.0*F*N)*invdetEFG
    s = math.sqrt(max(0.0, Kmean**2 - Kgauss))
    Kmin = Kgauss - s
    Kmax = Kgauss + s
    return n, Kmin, Kmax, Kgauss, Kmean
#############################################
