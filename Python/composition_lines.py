from PIL import Image, ImageDraw
import numpy
import sys


args = sys.argv

if len(args) < 2:
    width = 512
else:
    width = max(1,int(args[1]))
if len(args) < 3:
    height = width
else:
    height = max(1,int(args[2]))



npx = width*height

image = Image.new('RGBA', (width,height), color=(255,255,255,0))


THIRDS = True#False#
DIAGONALS = False#True
HALVES = False#True
TRIANGLES = True#False#
SPIRAL = True

clr = [(255,0,0,255),
       (0,255,0,255),
       (0,0,255,128),
       (0,255,0,255),
       (160,160,0,255)]
lwd = 1


draw = ImageDraw.Draw(image)

if THIRDS:
    for k in range(2):
        i = int((k+1)*width/3)
        j = int((k+1)*height/3)
        draw.line((i,0,i,height), fill=clr[0], width=lwd)
        draw.line((0,j,width,j), fill=clr[0], width=lwd)

if DIAGONALS:
    draw.line((0,0,width,height), fill=clr[1], width=lwd)
    draw.line((0,height,width,0), fill=clr[1], width=lwd)

if HALVES:
    i = int(width/2)
    j = int(height/2)
    draw.line((i,0,i,height), fill=clr[2], width=lwd)
    draw.line((0,j,width,j), fill=clr[2], width=lwd)

if TRIANGLES:
    x = (width*height**2)/(width**2 + height**2)
    y = height*x/width
    x = int(x)
    y = int(y)
    draw.line((0,0,width,height), fill=clr[3], width=lwd)
    draw.line((0,height,x,y), fill=clr[3], width=lwd)
    draw.line((width,0,width-x,height-y), fill=clr[3], width=lwd)

if SPIRAL:
    w = width
    h = height
    phi = 0.5*(1 + numpy.sqrt(5))#float(width)/float(height)#
    invphi = 1.0/phi
    turns = 3
    
    x = 0
    y = 0

    
    X = 1
    Y = 0
    for i in range(4*turns):
        d = invphi**(i+1)
        if (i+1)%4 < 2:
            sX = 1
        else:
            sX = -1
        if i%4 < 2:
            sY = 1
        else:
            sY = -1
        X += sX*d
        Y += sY*d
    X *= height
    Y *= height
    """
    R = 3
    draw.ellipse([X-R,Y-R,X+R,Y+R], fill=clr[4])
    """

    Xtarget = X#int(2*width/3)
    Ytarget = Y#int(2*height/3)

    x += Xtarget - X
    y += Ytarget - Y
    
    r = height
    for i in range(turns):
        draw.arc([x,y,x+2*r,y+2*r], 180, 270, fill=clr[4])

        x += r
        r *= invphi
        x -= r
        draw.arc([x,y,x+2*r,y+2*r], 270, 0, fill=clr[4])

        x += 2*r
        y += r
        r *= invphi
        x -= 2*r
        y -= r
        draw.arc([x,y,x+2*r,y+2*r], 0, 90, fill=clr[4])

        x += r
        y += 2*r
        r *= invphi
        x -= r
        y -= 2*r
        draw.arc([x,y,x+2*r,y+2*r], 90, 180, fill=clr[4])

        y += r
        r *= invphi
        y -= r
    
del draw

image.show()

