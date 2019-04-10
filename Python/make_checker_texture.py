import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

# arguments: nom_script, image_size(pixels), N, nom_fichier, dpi
args = sys.argv
if len(args) < 2:
    siz = 800
else:
    siz = int(args[1])

if len(args) < 3:
    N = 8
else:
    N = int(args[2])

if len(args) < 4:
    fname = 'checker.png'
else:
    fname = args[3]

if len(args) < 5:
    dpi = 72
else:
    dpi = int(args[4])


siz = siz/dpi#inches


# Checker texture
s = 1.0/float(N)

fig = plt.figure()
fig.set_size_inches(siz, siz)
ax = plt.Axes(fig, [0., 0., 1, 1])
ax.set_axis_off()
fig.add_axes(ax)

for i in range(N):
    for j in range(N):
        if (i+j)%2 == 0:
            r = patches.Rectangle((i*s,j*s),s,s,
                                  facecolor='k',
                                  linewidth=0)
        ax.add_patch(r) 

ax.set_aspect('equal', adjustable='box')
plt.savefig(fname, dpi=dpi)

    
