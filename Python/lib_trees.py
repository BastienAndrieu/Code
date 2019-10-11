class TreeNode:
    def __init__(self, extents, parent=None, ipoints=None):
        self.extents = extents #[xmin, xmax, ymin, ymax, ..]
        
        self.parent = parent
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        
        if ipoints is None:
            self.ipoints = []
        else:
            self.ipoints = ipoints
        
        self.children = []
        return
    #
    def is_leaf(self):
        return (len(self.children) < 1)
    #
    def split(self, points=[]):
        dim = len(self.extents)/2
        #
        aux = [(self.extents[2*i],
                0.5*(self.extents[2*i] + self.extents[2*i+1]),
                self.extents[2*i+1]) for i in range(dim)]
        #
        if dim == 2:
            # 4 children
            ext_child = [
                (aux[0][0], aux[0][1], aux[1][0], aux[1][1]), # SW
                (aux[0][1], aux[0][2], aux[1][0], aux[1][1]), # SE
                (aux[0][0], aux[0][1], aux[1][1], aux[1][2]), # NW
                (aux[0][1], aux[0][2], aux[1][1], aux[1][2]), # NE
            ]
        else:
            error('split: dim != 2 not supported yet')
        #
        for ext in ext_child:
            ipts_child = self.ipoints[:]
            for i in self.ipoints:
                for j in range(dim):
                    if points[i][j] < ext[2*j] or points[i][j] > ext[2*j+1]:
                        ipts_child.remove(i)
                        break
            self.children.append(
                TreeNode(
                    extents=ext,
                    parent=self,
                    ipoints=ipts_child
                )
            )
        return

