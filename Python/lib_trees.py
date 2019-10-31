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
    #
    def get_morton_code(self, tree_depth, tree_extents):
        dim = len(self.extents)/2
        mid = [0.5*(self.extents[2*i] + self.extents[2*i+1]) for i in range(dim)]
        #
        n = 2**tree_depth
        for i in range(dim):
            h = (tree_extents[2*i+1] - tree_extents[2*i])/n
            mid[i] = int((mid[i] - tree_extents[2*i])/h)
        #
        if dim == 2:
            m = interleave2(mid[0], mid[1])
        elif dim == 3:
            m = interleave3(mid[0], mid[1], mid[2])
        else:
            print 'get_morton_code: dim must be 2 or 3'
            exit()
        return m



def get_max_depth(node, depth):
    if node.is_leaf():
        depth = max(depth, node.depth)
    else:
        for child in node.children:
            depth = get_max_depth(child, depth)
    return depth

    
def flatten_tree(node, flat_tree, tree_depth, tree_extents):
    #print node.depth, node.is_leaf()
    if node.is_leaf():
        m = node.get_morton_code(tree_depth, tree_extents)
        #
        if m in flat_tree:
            print 'flatten_tree: collision'
            exit()
        #
        flat_tree[m] = node
    else:
        for child in node.children:
            flat_tree = flatten_tree(child, flat_tree, tree_depth, tree_extents)
    return flat_tree
        


#################################
# MORTON CODE
# http://code.activestate.com/recipes/577558-interleave-bits-aka-morton-ize-aka-z-order-curve/
#################################
def part1by1(n):
    n&= 0x0000ffff
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def unpart1by1(n):
    n&= 0x55555555
    n = (n ^ (n >> 1)) & 0x33333333
    n = (n ^ (n >> 2)) & 0x0f0f0f0f
    n = (n ^ (n >> 4)) & 0x00ff00ff
    n = (n ^ (n >> 8)) & 0x0000ffff
    return n


def interleave2(x, y):
    return part1by1(x) | (part1by1(y) << 1)


def deinterleave2(n):
    return unpart1by1(n), unpart1by1(n >> 1)


def part1by2(n):
    n&= 0x000003ff
    n = (n ^ (n << 16)) & 0xff0000ff
    n = (n ^ (n <<  8)) & 0x0300f00f
    n = (n ^ (n <<  4)) & 0x030c30c3
    n = (n ^ (n <<  2)) & 0x09249249
    return n


def unpart1by2(n):
    n&= 0x09249249
    n = (n ^ (n >>  2)) & 0x030c30c3
    n = (n ^ (n >>  4)) & 0x0300f00f
    n = (n ^ (n >>  8)) & 0xff0000ff
    n = (n ^ (n >> 16)) & 0x000003ff
    return n
    

def interleave3(x, y, z):
    return part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)


def deinterleave3(n):
    return unpart1by2(n), unpart1by2(n >> 1), unpart1by2(n >> 2)
