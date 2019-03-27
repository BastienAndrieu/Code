import numpy
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

pth = '/d/bandrieu/GitHub/Code/Python/Blender/'


siz = 10
lw = 3
cl = [1,0,0]

"""
for imesh in range(2):
    segments = numpy.loadtxt(pth+'intersection_segments_uv_'+str(imesh)+'.dat')
    
    fig = plt.figure()
    fig.set_size_inches(siz, siz)
    ax = plt.Axes(fig, [0., 0., 1, 1])

    ax.set_axis_off()
    fig.add_axes(ax)
    
    for s in segments:
        ax.plot(s[[0,2]], s[[1,3]], color=cl, linewidth=lw)
    
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(pth+'intersection_trace_'+str(imesh)+'.png', dpi=100)
"""


for imesh in range(2):
    segments = numpy.loadtxt(pth+'intersection_segments_uv_'+str(imesh)+'.dat')

    uvlayout = Image.open(pth+'uvlayout_'+str(imesh)+'.png')
    trace = Image.new("RGBA", uvlayout.size, (255,255,255,0))

    draw = ImageDraw.Draw(trace)
    for s in segments:
        x = s[[0,2]]*uvlayout.size[0]
        y = (1.0 - s[[1,3]])*uvlayout.size[1]
        draw.line((x[0], y[0], x[1], y[1]), fill=(0,0,0,255), width=lw)
    #trace.show()
    trace.save(pth+'intersection_trace_'+str(imesh)+'.png', 'PNG')
