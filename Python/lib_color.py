import numpy as np
import math

def sample_colormap(name, n):
    cm0 = np.loadtxt('/d/bandrieu/GitHub/These/memoire/figures/code/colormaps/'+name+'.dat')
    t0 = np.linspace(0,1,cm0.shape[0])
    t = np.linspace(0,1,n)
    cm = np.zeros((n,3))
    for j in range(3):
        cm[:,j] = np.interp(t, t0, cm0[:,j])
    return cm

##################################################

def rgb2hsv(rgb):
    if isinstance(rgb, (tuple,list)):
        rgb = np.asarray(rgb)
    hsv = []
    for c in rgb:
        r, g, b = c
        mx = max(r, g, b)
        mn = min(r, g, b)
        df = mx - mn
        if mx == mn:
            h = 0
        elif mx == r:
            h = (60 * ((g-b)/df) + 360) % 360
        elif mx == g:
            h = (60 * ((b-r)/df) + 120) % 360
        elif mx == b:
            h = (60 * ((r-g)/df) + 240) % 360
        if mx == 0:
            s = 0
        else:
            s = df/mx
        v = mx
        hsv.append([h/360.,s,v])
    return np.asarray(hsv)

##################################################

def hsv2rgb(hsv):
    if isinstance(hsv, (tuple,list)):
        hsv = np.asarray(hsv)
    rgb = []
    for c in hsv:
        h, s, v = c
        h = 360.*float(h)
        s = float(s)
        v = float(v)
        h60 = h / 60.0
        h60f = math.floor(h60)
        hi = int(h60f) % 6
        f = h60 - h60f
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        r, g, b = 0, 0, 0
        if hi == 0: r, g, b = v, t, p
        elif hi == 1: r, g, b = q, v, p
        elif hi == 2: r, g, b = p, v, t
        elif hi == 3: r, g, b = p, q, v
        elif hi == 4: r, g, b = t, p, v
        elif hi == 5: r, g, b = v, p, q
        rgb.append([r,g,b])
    return np.asarray(rgb)

##################################################

def cc_hsv(rgb, th=0, fs=1, fv=1):
    hsv = rgb2hsv(rgb)
    cc = []
    for c in hsv:
        h, s, v = c
        h = (h+th)%1
        s = max(0.0, min(1.0, fs*s))
        v = max(0.0, min(1.0, fv*v))
        cc.append([h,s,v])
    cc = hsv2rgb(np.asarray(cc))
    return cc
