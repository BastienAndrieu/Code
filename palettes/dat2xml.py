import sys
import numpy
import os

"""
args = sys.argv
imagename = args[1]
colormapname = args[2]
"""


files = os.popen('ls *.dat').readlines()

fout = open('my_colormaps.xml','w')

fout.write('<ColorMaps>\n')
for fname in files:
    print fname[:-1]
    fout.write('<ColorMap name="'+ fname[:-5] + '" space="RGB">\n')
    cmap = numpy.loadtxt(fname[:-1])
    x = numpy.linspace(0,1,len(cmap))
    for i, rgb in enumerate(cmap):
        fout.write(' <Point x="%.8s" o="1" r="%.8s" g="%.8s" b="%.8s"/>\n' %(x[i], rgb[0], rgb[1], rgb[2]))
    fout.write('</ColorMap>\n')
fout.write('</ColorMaps>')
fout.close()

