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
        if dim == 1:
            # 2 children
            ext_child = [
                (aux[0][0], aux[0][1]), # L
                (aux[0][1], aux[0][2])  # R
            ]
        elif dim == 2:
            # 4 children
            ext_child = [
                (aux[0][0], aux[0][1], aux[1][0], aux[1][1]), # SW
                (aux[0][1], aux[0][2], aux[1][0], aux[1][1]), # SE
                (aux[0][0], aux[0][1], aux[1][1], aux[1][2]), # NW
                (aux[0][1], aux[0][2], aux[1][1], aux[1][2]), # NE
            ]
        elif dim == 3:
            # 8 children
            ext_child = [
                (aux[0][0], aux[0][1], aux[1][0], aux[1][1], aux[2][0], aux[2][1]), # SWN
                (aux[0][0], aux[0][1], aux[1][0], aux[1][1], aux[2][1], aux[2][2]), # SWZ
                (aux[0][0], aux[0][1], aux[1][1], aux[1][2], aux[2][0], aux[2][1]), # SEN
                (aux[0][0], aux[0][1], aux[1][1], aux[1][2], aux[2][1], aux[2][2]), # SEZ
                (aux[0][1], aux[0][2], aux[1][0], aux[1][1], aux[2][0], aux[2][1]), # NWN
                (aux[0][1], aux[0][2], aux[1][0], aux[1][1], aux[2][1], aux[2][2]), # NWZ
                (aux[0][1], aux[0][2], aux[1][1], aux[1][2], aux[2][0], aux[2][1]), # NEN
                (aux[0][1], aux[0][2], aux[1][1], aux[1][2], aux[2][1], aux[2][2]), # NEZ
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
        
def place_points_in_tree(node, points, max_points_per_leaf=1):
    if len(node.ipoints) > max_points_per_leaf:
        node.split(points)
        for child in node.children:
            place_points_in_tree(child, points, max_points_per_leaf)
    return

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


##############################################
def min_dist2_box(coords, extents):
    dim = len(coords)
    assert len(extents) >= 2*dim, 'len(extents) < 2*len(coords)'
    inbox = 0
    min_dist2 = 0
    for i in range(dim):
        if coords[i] > extents[2*i+1]:
            _min_dist2 = coords[i] - extents[2*i+1]
            min_dist2 += _min_dist2 * _min_dist2
        elif coords[i] < extents[2*i]:
            _min_dist2 = coords[i] - extents[2*i]
            min_dist2 += _min_dist2 * _min_dist2
        else:
            inbox += 1
    inbox = (inbox == dim)
    return min_dist2, inbox


def max_dist2_box(coords, extents):
    dim = len(coords)
    assert len(extents) >= 2*dim, 'len(extents) < 2*len(coords)'
    inbox = 0
    max_dist2 = 0
    for i in range(dim):
        if coords[i] > extents[2*i+1]:
            _max_dist2 = coords[i] - extents[2*i]
            max_dist2 += _max_dist2 * _max_dist2
        elif coords[i] < extents[2*i]:
            _max_dist2 = coords[i] - extents[2*i+1]
            max_dist2 += _max_dist2 * _max_dist2
        else:
            val1 = coords[i] - extents[2*i]
            val2 = coords[i] - extents[2*i+1]
            max_dist2 += max(val1 * val1, val2 * val2)
            inbox += 1
    inbox = (inbox == dim)
    return max_dist2, inbox
##############################################
import numpy
def closest_point_in_cloud(cloud_pts, query_pts, max_points_per_leaf=1):
    coords_min = numpy.amin(cloud_pts, axis=0)
    coords_max = numpy.amax(cloud_pts, axis=0)

    dim = len(cloud_pts[0])
    
    ncloud = len(cloud_pts)
    extents = numpy.array(sum(zip(coords_min, coords_max), ()))

    root = TreeNode(
        extents=extents,
        ipoints=[i for i in range(ncloud)]
    )

    place_points_in_tree(
        node=root,
        points=cloud_pts,
        max_points_per_leaf=max_points_per_leaf)

    nquery = len(query_pts)
    closest_cloud_pt_id = [None]*nquery
    HUGE_VAL = 1e9*extents.dot(extents)
    closest_cloud_pt_dist2 = HUGE_VAL*numpy.ones(nquery)

    for i, coords in enumerate(query_pts):
        closest_cloud_pt_id[i], closest_cloud_pt_dist2[i] = recursion_tree_closest_point(
            coords,
            cloud_pts,
            root,
            closest_cloud_pt_id[i],
            closest_cloud_pt_dist2[i]
        )
    return closest_cloud_pt_id, closest_cloud_pt_dist2, root




def recursion_tree_closest_point(
        coords,
        cloud_pts,
        node,
        closest_pt_id,
        closest_pt_dist2
):
    min_dist2, inbox = min_dist2_box(coords, node.extents)
    #
    if min_dist2 <= closest_pt_dist2 or inbox:
        if node.is_leaf():
            for ipt in node.ipoints:
                v = coords - cloud_pts[ipt]
                point_dist2 = v.dot(v)
                if point_dist2 < closest_pt_dist2:
                    closest_pt_id = ipt
                    closest_pt_dist2 = point_dist2
        else:
            for child in node.children:
                if len(child.ipoints) > 0:
                    closest_pt_id, closest_pt_dist2 = recursion_tree_closest_point(
                        coords,
                        cloud_pts,
                        child,
                        closest_pt_id,
                        closest_pt_dist2
                    )
            """
            n_selec = 0
            dist_child = 1e9*numpy.ones(len(node.children))
            sort_child = [None]*len(node.children)
            inbox_child = numpy.zeros(len(node.children), dtype=bool)
            for ichild, child in enumerate(node.children):
                child_min_dist2, child_inbox = min_dist2_box(coords, child.extents)
                i1 = n_selec
                while i1 > 0 and dist_child[i1-1] > child_min_dist2:
                    dist_child[i1] = dist_child[i1-1]
                    sort_child[i1] = sort_child[i1-1]
                    inbox_child[i1] = inbox_child[i1-1]
                    i1 -= 1
                sort_child[i1] = ichild
                dist_child[i1] = child_min_dist2
                inbox_child[i1] = child_inbox
                n_selec += 1

            for j in range(n_selec):
                j1 = n_selec - 1 - j
                child = node.children[sort_child[j1]]
                if len(child.ipoints) > 0:
                    closest_pt_id, closest_pt_dist2 = recursion_tree_closest_point(
                        coords,
                        cloud_pts,
                        child,
                        closest_pt_id,
                        closest_pt_dist2
                    )
            """
        #
    return closest_pt_id, closest_pt_dist2





def plot_tree(node, ax, show_depth=False, color='0.8', linestyle='-', linewidth=1):
    if node.is_leaf():
        x = [node.extents[0], node.extents[1], node.extents[1], node.extents[0], node.extents[0]]
        y = [node.extents[2], node.extents[2], node.extents[3], node.extents[3], node.extents[2]]
        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth)
        if show_depth:
            x, y = [0.5*(node.extents[2*i] + node.extents[2*i+1]) for i in range(2)]
            ax.text(x, y, str(node.depth), color='r')
    else:
        for child in node.children:
            plot_tree(child, ax, show_depth, color, linestyle, linewidth)
    return 
