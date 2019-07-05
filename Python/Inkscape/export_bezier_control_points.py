#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inkex, simplestyle, simplepath
import logging
from math import hypot

SPT = 1e-4 # Same-Point Tolerance

#########################################
def distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])
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
def write_path_to_stream(pathdata, stream):
    myPaths = get_complex_path_bp(pathdata)
    if myPaths is None:
        return
    for bp in myPaths:
        nbp = len(bp)/2
        stream.write('%d\n' % nbp)
        for i in range(nbp):
            stream.write('%s %s\n' % (bp[2*i], bp[2*i+1]))
    return
#########################################
class ExportBCP(inkex.Effect):
    def __init__(self):
        inkex.Effect.__init__(self)
        self._set_up_options()

    def _set_up_options(self):
        parser = self.OptionParser
        parser.add_option(
            "-o",
            "--output",
            action="store",
            type="string",
            dest="outputfile",
            default=None,
            help=""
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
    def effect(self):
        nodes = self.selected_sorted

        fout = open(self.options.outputfile, 'w')
        
        for node in nodes:
            pathdata = None
            if node.tag == _ns('path'):
                pathdata = self._handle_path(node)
                
                write_path_to_stream(pathdata, fout)
            else:
                continue

        fout.close()

# Create effect instance and apply it.
effect = ExportBCP()
effect.affect()
