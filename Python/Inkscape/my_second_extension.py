#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inkex, simplestyle, simplepath
import logging
from math import hypot

SPT = 1e-4 # Same-Point Tolerance
EPS_VEC = 1e-8 # Zero-Vector magnitude

#########################################
def distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])
#########################################
def tangent_to_normal(tng):
    mag = hypot(tng[0], tng[1])
    if mag < EPS_VEC:
        return (0,0)
    else:
        invmag = 1.0/mag
        return (-tng[1]*invmag, tng[0]*invmag)
#########################################
def _ns(element_name, name_space='svg'):
    """
    copied from tikz_export.py (svg2tikz extension)
    """
    return inkex.addNS(element_name, name_space)
#########################################
def get_complex_path_bp(pathdata):
    if not pathdata: return None
    myPaths = []
    bp = []
    for cmd, params in pathdata:
        if cmd == 'M':
            bp = [x for x in params]
        elif cmd == 'Z':
            bptmp = [x for x in params]
            if distance(bp[-2:], myPaths[0][0:2]) > SPT:
                if len(bptmp) > 1:
                    bp.extend(bptmp)
                else:
                    bp.extend(myPaths[0][0:2])
                myPaths.append(bp)
            bp = []
        else:
            bp.extend([x for x in params])
            myPaths.append(bp)
            bp = bp[-2:]
    return myPaths
#########################################
def eval_line(bp, t):
    s = 1 - t
    x = s*bp[0] + t*bp[2]
    y = s*bp[1] + t*bp[3]
    return (x, y)
#########################################
def eval_quadratic_bezier(bp, t):
    s = 1 - t
    s2 = s**2
    t2 = t**2
    x = s2*bp[0] + 2*s*t*bp[2] + t2*bp[4]
    y = s2*bp[1] + 2*s*t*bp[3] + t2*bp[5]
    return (x,y)
#########################################
def eval_cubic_bezier(bp, t):
    s = 1 - t
    s2 = s**2
    s3 = s**3
    t2 = t**2
    t3 = t**3
    x = s3*bp[0] + 3*s2*t*bp[2] + 3*s*t2*bp[4] + t3*bp[6]
    y = s3*bp[1] + 3*s2*t*bp[3] + 3*s*t2*bp[5] + t3*bp[7]
    return (x,y)
#########################################
def eval_tangent_line(bp, t):
    dx = bp[2] - bp[0]
    dy = bp[3] - bp[1]
    return (dx, dy)
#########################################
def eval_tangent_quadratic_bezier(bp, t):
    dbp = [
        2*(bp[2] - bp[0]),
        2*(bp[3] - bp[1]),
        2*(bp[4] - bp[2]),
        2*(bp[5] - bp[3])]
    return eval_line(dbp, t)
#########################################
def eval_tangent_cubic_bezier(bp, t):
    dbp = [
        3*(bp[2] - bp[0]),
        3*(bp[3] - bp[1]),
        3*(bp[4] - bp[2]),
        3*(bp[5] - bp[3]),
        3*(bp[6] - bp[4]),
        3*(bp[7] - bp[5])]
    return eval_quadratic_bezier(dbp, t)
#########################################
def t_global_to_local(tglob, npaths=1):
    EPS = 1e-4
    if npaths <= 1:
        return tglob, 0
    else:
        dt = 1.0/npaths
    if tglob > 1 - EPS:
        return 1.0, npaths-1
    elif tglob < EPS:
        return 0.0, 0
    else:
        for ipath in range(npaths):
            if tglob < (ipath+1)*dt:
                return tglob/dt - ipath, ipath
#########################################
class EvalPointOnPath(inkex.Effect):
    def __init__(self):
        inkex.Effect.__init__(self)
        self._set_up_options()

    def _set_up_options(self):
        parser = self.OptionParser
        parser.add_option(
            '-t',
            '--tvalue',
            action = 'store',
            type = 'float',
            dest = 'tvalue',
            default = 0.5,
            help = 'Parameter value'
        )
        parser.add_option(
            '-s',
            '--scale',
            action = 'store',
            type = 'float',
            dest = 'scale',
            default = 1.0,
            help = 'Normal vector scale'
        )
    #######################
    def getselected(self):
        """Get selected nodes in document order

        The nodes are stored in the selected dictionary and as a list of
        nodes in selected_sorted.
        copied from tikz_export.py (svg2tikz extension)
        """
        self.selected_sorted = []
        self.selected = {}
        if len(self.options.ids) == 0:
            return
            # Iterate over every element in the document
        for node in self.document.getiterator():
            node_id = node.get('id', '')
            if node_id in self.options.ids:
                self.selected[node_id] = node
                self.selected_sorted.append(node)
    #######################
    def _handle_path(self, node):
        """
        adapted from tikz_export.py (svg2tikz extension)
        """
        try:
            raw_path = node.get('d')
            p = simplepath.parsePath(raw_path)
        except:
            logging.warning('Failed to parse path %s, will ignore it', raw_path)
            p = None
        return p
    #######################
    def transform(self, coord_list, cmd=None):
        """
        Apply transformations to input coordinates
        copied from tikz_export.py (svg2tikz extension)
        """
        coord_transformed = []
        # TEMP:
        if cmd == 'Q':
            return tuple(coord_list)
        try:
            if not len(coord_list) % 2:
                for x, y in nsplit(coord_list, 2):
                    #coord_transformed.append("%.4fcm" % ((x-self.x_o)*self.x_scale))
                    #oord_transformed.append("%.4fcm" % ((y-self.y_o)*self.y_scale))
                    coord_transformed.append("%.4f" % x)
                    coord_transformed.append("%.4f" % y)
            elif len(coord_list) == 1:
                coord_transformed = ["%.4fcm" % (coord_list[0] * self.x_scale)]
            else:
                coord_transformed = coord_list
        except:
            coord_transformed = coord_list
        return tuple(coord_transformed)
    #######################


    def _eval_complex_path(self, pathdata):
        myPaths = get_complex_path_bp(pathdata)
        if myPaths is None: return None
        tglob = self.options.tvalue
        tloc, ipath = t_global_to_local(tglob, len(myPaths))
        bp = myPaths[ipath]
        if len(bp) == 4:
            xy = eval_line(bp, tloc)
        elif len(bp) == 6:
            xy = eval_quadratic_bezier(bp, tloc)
        elif len(bp) == 8:
            xy = eval_cubic_bezier(bp, tloc)
        else:
            xy = None
        return (xy, tloc, ipath)
    #######################
    def _eval_normal_complex_path(self, pathdata):
        myPaths = get_complex_path_bp(pathdata)
        if myPaths is None: return None
        tglob = self.options.tvalue
        tloc, ipath = t_global_to_local(tglob, len(myPaths))
        bp = myPaths[ipath]
        if len(bp) == 4:
            tng = eval_tangent_line(bp, tloc)
        elif len(bp) == 6:
            tng = eval_tangent_quadratic_bezier(bp, tloc)
        elif len(bp) == 8:
            tng = eval_tangent_cubic_bezier(bp, tloc)
        else:
            return None
        return tangent_to_normal(tng)
    #######################
    def print_tvalue_at_xy(self, txt=None, xy=(0,0)):
        svg = self.document.getroot()
        width  = self.unittouu(svg.get('width'))
        height = self.unittouu(svg.attrib['height'])
        layer = inkex.etree.SubElement(svg, 'g')
        layer.set(inkex.addNS('label', 'inkscape'), 'xy Layer')
        layer.set(inkex.addNS('groupmode', 'inkscape'), 'layer')
        text = inkex.etree.Element(inkex.addNS('text','svg'))
        x, y = xy
        if txt is None:
            text.text = '%.4f' % (self.options.tvalue)
        else:
            text.text = txt
        text.set('x', str(xy[0]))
        text.set('y', str(xy[1]))
        style = {'text-align' : 'left', 'text-anchor' : 'left'}
        text.set('style', simplestyle.formatStyle(style))
        layer.append(text)
    #######################
    def draw_vector_at_xy(self, xy=(0,0), uv=(0,0), scale=1):
        svg = self.document.getroot()
        layer = inkex.etree.SubElement(svg, 'g')
        layer.set(inkex.addNS('label', 'inkscape'), 'uv Layer')
        layer.set(inkex.addNS('groupmode', 'inkscape'), 'layer')
        x, y = xy
        u, v = uv
        line_style   = {
            'stroke': '#FF0000',
            'stroke-width': '1',
            'fill': 'none'
        }
        vector_attribs = {
            'style' : simplestyle.formatStyle(line_style),
            'd' : 'M '+str(x)+','+str(y)+' l '+str(scale*u)+','+str(scale*v)
        }
        vector = inkex.etree.Element(inkex.addNS('path','svg'), vector_attribs)
        layer.append(vector)

    #######################
    def effect(self):
        nodes = self.selected_sorted

        for node in nodes:
            pathdata = None
            if node.tag == _ns('path'):
                pathdata = self._handle_path(node)

                if False:
                    k = 0
                    for cmd, params in pathdata:
                        self.print_tvalue_at_xy(cmd, (20,20+k*60))
                        self.print_tvalue_at_xy(str(params), (20,40+k*60))
                        k += 1

                if False:
                    paths = get_complex_path_bp(pathdata)
                    for path in paths:
                        self.print_tvalue_at_xy(str(path), (20,60+k*60))
                        k += 1
                
                
                xy, tloc, ipath = self._eval_complex_path(pathdata)
                uv = self._eval_normal_complex_path(pathdata)
                
                if xy is not None:
                    #self.print_tvalue_at_xy(txt='%.4f , %d' %(tloc, ipath), xy=xy)
                    self.draw_vector_at_xy(
                        xy=xy,
                        uv=uv,
                        scale=self.options.scale)
            else:
                continue

            
                


# Create effect instance and apply it.
effect = EvalPointOnPath()
effect.affect()
