import numpy as np
##########################################################
def norm2(a):
    return np.sqrt(np.sum(np.power(a,2)))
##########################################################
def angle3d(u,v):
    return np.arctan2(norm2(np.cross(u,v)), np.dot(u,v))
##########################################################
def one_parameter_EoS(g, dg, d2g, r, dr, v):
    m = g.shape[1]
    n = len(v)-1
    e = np.zeros((m,n,3))
    vmin = np.amin(v)
    vmax = np.amax(v)
    v = 2.0*np.pi*(v - vmin)/(vmax - vmin)
    cv = np.cos(v)
    sv = np.sin(v)
    for i in range(m):
        normdg = norm2(dg[:,i])
        sina = dr[i]/normdg
        occ = g[:,i] - r[i]*sina*dg[:,i]/normdg
        rcc = r[i]*np.sqrt(1. - sina**2)
        bw = dg[:,i]/normdg
        bu = d2g[:,i] - np.dot(d2g[:,i],bw)*bw
        bu = bu/norm2(bu)
        bv = np.cross(bw,bu)
        for j in range(n):
            e[i,j] = occ + rcc*(cv[j]*bu + sv[j]*bv)
    return e
##########################################################
def two_parameter_EoS(s, dus, dvs, r, dur, dvr):
    m = s.shape[1]
    n = s.shape[2]
    e = np.zeros((2,m,n,3))
    for j in range(n):
        for i in range(m):
            nor = np.cross(dus[:,i,j],dvs[:,i,j])
            nor = nor/norm2(nor)
            E = np.dot(dus[:,i,j], dus[:,i,j])
            F = np.dot(dus[:,i,j], dvs[:,i,j])
            G = np.dot(dvs[:,i,j], dvs[:,i,j])
            w = (dur[i,j]*G - dvr[i,j]*F)*dus[:,i,j] + (dvr[i,j]*E - dur[i,j]*F)*dvs[:,i,j]
            w /= F**2 - E*G
            sqrt1mw2 = np.sqrt(1. - np.sum(np.power(w,2)))
            for k in range(2):
                e[k,i,j] = s[:,i,j] + r[i,j]*(w + sqrt1mw2*nor*(-1.)**k)
    return e
##########################################################
def trimmed_canal_surface(g, el, er, dg, r, dr, v):
    m = g.shape[1]
    n = len(v)
    e = np.zeros((m,n,3))
    vmin = np.amin(v)
    vmax = np.amax(v)
    v = (v - vmin)/(vmax - vmin)
    for i in range(m):
        normdg = norm2(dg[:,i])
        sina = dr[i]/normdg
        occ = g[:,i] - r[i]*sina*dg[:,i]/normdg
        bw = -dg[:,i]/normdg
        bu = er[:,i]/norm2(er[:,i])
        bv = np.cross(bw,bu)
        ang = angle3d(er[:,i],el[:,i])
        for j in range(m):
            e[i,j] = g[:,i] + r[i]*(np.cos(v[j]*ang)*bu + np.sin(v[j]*ang)*bv)
    return e
##########################################################
