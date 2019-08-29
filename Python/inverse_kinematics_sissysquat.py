import numpy
from numpy import array, cos, sin, arccos, pi, radians, arctan2

import matplotlib.pyplot as plt
from matplotlib import animation

###################################
# PARAMETERS
a_shin = 0.5*pi
l_strap = 0.8#0.7
l_shin = 0.55
l_thigh = 0.48
l_trunk = 0.55
l_arm = 0.3
l_forearm = 0.35

###########
rescale = 1.6/(l_shin + l_thigh + l_trunk)
l_shin *= rescale
l_thigh *= rescale
l_trunk *= rescale
l_arm *= rescale
l_forearm *= rescale
###########

xy_bar = array([0.6, 2.0])

aabb = [[-1.5,0], [1,xy_bar[1]]]
###################################


xy_toes = [0.2,0]
r_head = 0.08
l_neck = 0.8*r_head






###################################
def norm2(a):
    return numpy.sqrt(a.dot(a))
###################################
def distance(a, b):
    return norm2(a - b)
###################################
def angle_between_vectors(u, v):
    a = arccos(min(1, max(-1, u.dot(u)/(norm2(u)*norm2(v)))))
    if u[0]*v[1] < u[1]*v[0]: a = -a
    return a
###################################

###################################
def plot_body(ax, xy_hip, xy_shoulder, xy_elbow, xy_wrist, color='b', linestyle='solid', marker='.', fill_head=False):
    v = (xy_shoulder - xy_hip)/distance(xy_shoulder, xy_hip)
    body = numpy.vstack([xy_toes, [0,0], xy_knee, xy_hip, xy_shoulder, xy_elbow, xy_wrist])#, xy_bar])
    ax.plot(body[:,0], body[:,1], color=color, linestyle=linestyle, marker=marker)
    v = (xy_shoulder - xy_hip)/distance(xy_shoulder, xy_hip)
    xy_neck = xy_shoulder + v*l_neck
    ax.plot([xy_shoulder[0], xy_neck[0]], [xy_shoulder[1], xy_neck[1]], color=color, linestyle=linestyle)
    xy_head = xy_neck + v*r_head
    ax.add_artist(
        plt.Circle(
            xy_head,
            r_head,
            ec=color,
            fc=color,
            fill=fill_head
        )
    )
    return
###################################
def plot_strap(ax, xy_wrist, color='b', linestyle='dashed'):
    strap = numpy.vstack([xy_wrist, xy_bar])
    ax.plot(strap[:,0], strap[:,1], color=color, linestyle=linestyle)
    return
###################################
def make_neck_head(xy_hip, xy_shoulder):
    v = (xy_shoulder - xy_hip)/distance(xy_shoulder, xy_hip)
    a = arctan2(v[1], v[0])
    t = a + numpy.linspace(-pi,pi,100)
    xy_neck = xy_shoulder + v*l_neck
    xy_head = xy_neck + v*r_head
    """
    xy = numpy.vstack([
        xy_shoulder,
        xy_neck,
        [[xy_head + r_head*array([cos(t[i]), sin(t[i])]) for i in range(len(t))]]
    ])
    """
    #xy = array([xy_head + r_head*array[cos(t[i]), sin(t[i])] for i in range(len(t))])
    xy = numpy.vstack([xy_shoulder, xy_neck])
    xy = numpy.vstack([xy, [xy_head + r_head*array([cos(u), sin(u)]) for u in t]])
    #for u in t:
    #    xy = numpy.vstack([xy, xy_head + r_head*array([cos(u), sin(u)])])
    return xy
###################################
def angles_to_positions(a_thigh, a_trunk, a_arm, a_forearm):
    a = a_shin + a_thigh
    xy_hip = xy_knee + l_thigh*array([cos(a), sin(a)])
    a += a_trunk
    xy_shoulder = xy_hip + l_trunk*array([cos(a), sin(a)])
    a += a_arm
    xy_elbow = xy_shoulder + l_arm*array([cos(a), sin(a)])
    a += a_forearm
    xy_wrist = xy_elbow + l_forearm*array([cos(a), sin(a)])
    return xy_hip, xy_shoulder, xy_elbow, xy_wrist
###################################

###################################
def positions_to_angles(xy_hip, xy_shoulder, xy_elbow, xy_wrist):
    a_thigh = angle_between_vectors(xy_hip - xy_knee, xy_knee)
    a_trunk = angle_between_vectors(xy_shoulder - xy_hip, xy_hip - xy_knee)
    a_arm = angle_between_vectors(xy_elbow - xy_shoulder, xy_shoulder - xy_hip)
    a_forearm = angle_between_vectors(xy_wrist - xy_elbow, xy_elbow - xy_shoulder)
    return a_thigh, a_trunk, a_arm, a_forearm
###################################


###################################
def IK(a_thigh, a_trunk, a_arm, a_forearm, fix_hip=False, iteration_max=30, tol=1e-2):
    # initial positions
    xy_hip, xy_shoulder, xy_elbow, xy_wrist = angles_to_positions(a_thigh, a_trunk, a_arm, a_forearm)
    #
    for iteration in range(iteration_max):
        d_wrist_bar = distance(xy_wrist, xy_bar)
        err = abs(d_wrist_bar - l_strap)
        if err < tol:
            # converged
            print 'converged in %d iterations' % iteration
            break
        print '\titer.#%d, err = %s' % (iteration, err)
        #
        # Forward reaching
        v = xy_wrist - xy_bar
        xy_wrist_F = xy_bar + v*l_strap/norm2(v)
        v = xy_elbow - xy_wrist_F
        xy_elbow_F = xy_wrist_F + v*l_forearm/norm2(v)
        v = xy_shoulder - xy_elbow_F
        xy_shoulder_F = xy_elbow_F + v*l_arm/norm2(v)
        if not fix_hip:
            v = xy_hip - xy_shoulder_F
            xy_hip_F = xy_shoulder_F + v*l_trunk/norm2(v)
        #
        # Backward reaching
        if not fix_hip:
            v = xy_hip_F - xy_knee
            xy_hip = xy_knee + v*l_thigh/norm2(v)
        v = xy_shoulder_F - xy_hip
        xy_shoulder = xy_hip + v*l_trunk/norm2(v)
        v = xy_elbow_F - xy_shoulder
        xy_elbow = xy_shoulder + v*l_arm/norm2(v)
        v = xy_bar - xy_elbow
        xy_wrist = xy_elbow + v*l_forearm/norm2(v)
    return xy_hip, xy_shoulder, xy_elbow, xy_wrist
###################################











###################################
xy_knee = l_shin*array([cos(a_shin), sin(a_shin)])
###################################

###################################
# input a_thigh (knee flexion angle)
# output all joints flexion angles and positions
instants = []
a_trunk = 0
a_arm = -0.9*pi
a_forearm = 0.8*pi
ninstants = 20
for a in numpy.linspace(0,radians(100),ninstants):
    xy_hip, xy_shoulder, xy_elbow, xy_wrist = IK(a_thigh=a, a_trunk=a_trunk, a_arm=a_arm, a_forearm=a_forearm, fix_hip=True)
    #a_thigh, a_trunk, a_arm, a_forearm = positions_to_angles(xy_hip, xy_shoulder, xy_elbow, xy_wrist)
    instants.append([xy_hip, xy_shoulder, xy_elbow, xy_wrist])

###################################
"""
fig, ax = plt.subplots()

for i in range(len(instants[0])):
    trajectory = array([body[i] for body in instants])
    ax.plot(trajectory[:,0], trajectory[:,1], '--')

cl = ['b', 'k']
for j, i in enumerate(range(ninstants)):#[0,-1]):
    xy_hip, xy_shoulder, xy_elbow, xy_wrist = instants[i]
    plot_body(ax, xy_hip, xy_shoulder, xy_elbow, xy_wrist, color=cl[j%len(cl)])
    plot_strap(ax, xy_wrist, color=cl[j%len(cl)])

ax.add_artist(
        plt.Circle(
            xy_bar,
            l_strap,
            ec='r',
            ls='dotted',
            fill=False
        )
    )
ax.plot(xy_bar[0], xy_bar[1], 'rs')

ax.set_aspect('equal')
ax.set_xlim([aabb[0][0], aabb[1][0]])
ax.set_ylim([aabb[0][1], aabb[1][1]])
plt.show()
"""
###################################



###################################
# ANIMATION
pause = 10
instants.extend([instants[-1] for i in range(pause)])
instants.extend([instants[i] for i in range(ninstants-1,-1,-1)])
instants.extend([instants[0] for i in range(pause)])


fig = plt.figure()
ax = plt.axes(xlim=(aabb[0][0], aabb[1][0]), ylim=(aabb[0][1], aabb[1][1]))
line_body, = ax.plot([], [], lw=2, color='b', marker='.')
line_strap, = ax.plot([], [], lw=1, color='k')
line_head, = ax.plot([], [], lw=2, color='b')

# initialization function: plot the background of each frame
def init():
    line_body.set_data([], [])
    line_strap.set_data([], [])
    line_head.set_data([], [])
    return line_body, line_strap, line_head,

# animation function.  This is called sequentially
def animate(i):
    xy_hip, xy_shoulder, xy_elbow, xy_wrist = instants[i]
    body = numpy.vstack([xy_toes, [0,0], xy_knee, xy_hip, xy_shoulder, xy_elbow, xy_wrist])
    line_body.set_data(body[:,0], body[:,1])
    #
    strap = numpy.vstack([xy_wrist, xy_bar])
    line_strap.set_data(strap[:,0], strap[:,1])
    #
    xy = make_neck_head(xy_hip, xy_shoulder)
    line_head.set_data(xy[:,0], xy[:,1])
    return line_body, line_strap, line_head,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=len(instants),
    interval=40,
    blit=True
)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=25, extra_args=['-vcodec', 'libx264'])

plt.show()
###################################
