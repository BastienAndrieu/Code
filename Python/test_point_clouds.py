import numpy
import time
import matplotlib.pyplot as plt

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python')
import lib_trees as lbt


dim = 3

nclusters = 5#50
nmax_per_cluster = 2000#200
noise = 0.3

point_clouds = []
for iptc in range(2):
    xyc = 2*numpy.random.rand(nclusters,dim) - 1
    xy = numpy.empty((0,dim), float)
    for c in xyc:
        ni = numpy.random.randint(nmax_per_cluster)
        xy = numpy.vstack([
            xy,
            numpy.tile(c, (ni,1)) + noise*(2*numpy.random.rand(ni,dim) - 1)
        ])
    point_clouds.append(xy)

#point_clouds[1] = point_clouds[1][:1]

for i in range(2):
    print 'point cloud #%d: %d point(s)' % (i, len(point_clouds[i]))

results = []

# TREE METHOD
start_time = time.time()
result_tree = []
roots = []
for iptc in range(2):
    pt_id, pt_dist2, root = lbt.closest_point_in_cloud(
        cloud_pts=point_clouds[(iptc+1)%2],
        query_pts=point_clouds[iptc],
        max_points_per_leaf=10,
    )
    #print '\n\n\n\n'
    result_tree.append([pt_id, pt_dist2])
    roots.append(root)
elapsed = time.time() - start_time
results.append(result_tree)
print '       Tree: %s s' % elapsed



comparison = max(len(point_clouds[0]), len(point_clouds[1])) < 3000
    
if comparison:
    # BRUTE FORCE
    HUGE_VAL = 1e9
    start_time = time.time()
    result_brute = []
    for iptc in range(2):
        nq = len(point_clouds[iptc])
        nc = len(point_clouds[(iptc+1)%2])
        pt_id = [None]*nq
        pt_dist2 = HUGE_VAL*numpy.ones(nq)
        for i, q in enumerate(point_clouds[iptc]):
            for j, p in enumerate(point_clouds[(iptc+1)%2]):
                dist2 = (p - q).dot(p - q)
                if dist2 < pt_dist2[i]:
                    pt_dist2[i] = dist2
                    pt_id[i] = j
                    result_brute.append([pt_id, pt_dist2])
                    elapsed = time.time() - start_time
                    print 'Brute Force: %s s' % elapsed

    results.append(result_brute)

    # COMPARE
    for iptc in range(2):
        nmismatch = 0
        for i in range(len(point_clouds[iptc])):
            if results[0][iptc][0][i] != results[1][iptc][0][i]:
                #print 'mismatch cloud#%d, pt#%d' % (iptc, i)
                nmismatch += 1
                print 'mismatches: %d/%d' % (nmismatch, len(point_clouds[iptc]))







if dim == 2:
    fig, axes = plt.subplots(2,2)
    for k in range(len(results)):
        for iptc in range(2):
            ax = axes[k][iptc]
            if k == 0:
                lbt.plot_tree(roots[iptc], ax, color='g', linestyle='-', linewidth=0.2)

            jptc = (iptc+1)%2
            ax.plot(point_clouds[iptc][:,0], point_clouds[iptc][:,1], 'r.')
            ax.plot(point_clouds[jptc][:,0], point_clouds[jptc][:,1], 'b.')        

            for i, xy in enumerate(point_clouds[iptc]):
                x, y = xy
                j = results[k][iptc][0][i]
                if j is not None:
                    u, v = point_clouds[jptc][j]
                    ax.plot([x, u], [y, v], 'k-', lw=0.2)
                else:
                    print iptc, i, j

            ax.set_aspect('equal')
    plt.show()
