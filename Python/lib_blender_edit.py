import bpy
import bmesh



### Triangulate an object
# to triangulate the active object, use:
# triangulate_object(bpy.context.active_object)
def triangulate_object(obj):
    me = obj.data
    # Get a BMesh representation
    bm = bmesh.new()
    bm.from_mesh(me)

    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()
    return



### Submesh extraction
def extract_selected_submesh(obj):
    # switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    me = obj.data
    # Get a BMesh representation
    bm = bmesh.from_edit_mesh(me)

    # get list of selected faces
    listfaces = [f for f in bm.faces if f.select]

    # re-order vertices of the selected submesh and extract local connectivity
    verts = {} # dictionary: global vertex index -> local vertex index
    xyz = []   # xyz coordinates of the submesh's vertices
    faces = [] # connectivity table of the submesh's faces (local indices)
    for f in listfaces:
        fv = []
        for v in f.verts:
            if v.index not in verts:
                verts[v.index] = len(verts)
                xyz.append(list(v.co))
            fv.append(verts[v.index])
        faces.append(fv)

    # leave edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # clone submesh in a separate object
    name = obj.name + "_submesh"
    submsh = bpy.data.meshes.new(name)
    subobj = bpy.data.objects.new(name, submsh)
    bpy.context.scene.objects.link(subobj)
    submsh.from_pydata(xyz,[],faces)
    submsh.update(calc_edges=True)
    
    bm.free()
    return subobj



### Laplacian smoothing
def laplacian_smoothing(obj, npasses=1):
    # switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    me = obj.data
    # Get a BMesh representation
    bm = bmesh.from_edit_mesh(me)

    # get list of free vertices (i.e. not on the boundary)
    freeverts = [v for v in bm.verts if not v.is_boundary]

    # get vertex-vertex adjacency lists
    neighbors = [[e.other_vert(v) for e in v.link_edges] for v in bm.verts]

    # perform plain Laplacian smoothing on free vertices
    for passe in range(npasses):
        for v in freeverts:
            cotmp = v.co.copy()
            for w in neighbors[v.index]:
                cotmp = cotmp + w.co
            cotmp = cotmp / float(len(neighbors[v.index]) + 1)
            v.co = cotmp
            
    # leave edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    bm.free()
    return
