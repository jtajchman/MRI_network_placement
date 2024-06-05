import os
import numpy as np
# Various mesh tools
# Editing simnibs meshes 
from simnibs.mesh_tools import mesh_io
# Converting between mesh filetypes
import meshio
# Mesh visualization
import pyvista as pv
from pyvista.plotting.plotter import Plotter

# run with: simnibs_python .\extract_normal.py
# requires simnibs to be installed and simnibs\bin to be added to environment variables

fn = 'Orig_mesh/MJ.msh' # Name of original mesh file
target = [-10.02, -22.03, 64.46] # IN WORLD COORDINATES in mm
target_radius = 5 # Radius to label around target in visualization
vis_option = 1 # Whether to visualize results

mesh = mesh_io.read_msh(fn)
wm_surface = mesh.crop_mesh(tags = 1001) # Extract white matter surface
gm_surface = mesh.crop_mesh(tags = 1002) # Extract gray matter surface

# Take simnibs mesh and convert to pyvista mesh via filetype conversion
def msh2pv(msh, fn, ftype='ply', overwrite=False):
    ftype_file = f'msh/{fn}.{ftype}'
    
    # If the file needs to be written
    if not os.path.exists(ftype_file) and not overwrite:
        msh_file = f'msh/{fn}.msh'
        msh.write(msh_file)
        meshio.read(msh_file).write(ftype_file)
    # Return the mesh as read by pyvista
    return pv.get_reader(ftype_file).read()

# Filetype to be used to communicate between gmsh and pyvista
ftype = 'ply' # ply, vtu, obj, stl | valid (error-free) filetypes in rough order of fastest to slowest (ply fastest by far)
gm_pv_mesh = msh2pv(gm_surface, 'GM', ftype)
wm_pv_mesh = msh2pv(wm_surface, 'WM', ftype)

# Get direction of a surface element (after mesh smoothing) based on element index
def get_smoothed_normal(surface, index, smooth):
    """
    Calculates smoothed normals on the layer.
    :return: Smoothed normal vectors.
    """
    return surface.triangle_normals(smooth=smooth).value[index]

# Extract data from the mesh with respect to a given target
def get_msh_data(msh, tar):
    # Coordinates of the baricenters of each surface element
    m_bar = msh.elements_baricenters()
    # Distance of each element from the target
    m_tar_dist = [np.linalg.norm(bar - tar) for bar in m_bar.value]
    # Index of element closest to target
    m_elm_index = np.argmin(m_tar_dist)
    # Baricenter position of element closest to target
    m_target_pos = m_bar.value[m_elm_index]
    return m_tar_dist, m_elm_index, m_target_pos

# Get coordinates of gm element closest to target (gm target)
gm_tar_dist, gm_elm_index, gm_target_pos = get_msh_data(gm_surface, target)
# Get coordinates of wm element closest to gm target to pair with it (wm pair/shortest-distance pair)
wm_pair_dist, wm_elm_index, wm_pair_pos = get_msh_data(wm_surface, gm_target_pos)

# Label the pyvista mesh to show a region around the target for visualization
def label_target_pv(msh, tar_dist, tar_radius, pv_msh):
    m_target_elm = msh.elm.elm_number[(np.array(tar_dist) < tar_radius)]
    m_target = np.zeros(msh.elm.nr)
    m_target[m_target_elm - 1] = 1
    pv_msh.cell_data['Target'] = m_target

label_target_pv(gm_surface, gm_tar_dist, target_radius, gm_pv_mesh)
label_target_pv(wm_surface, wm_pair_dist, target_radius, wm_pv_mesh)

# Get normal direction of gm surface at gm target
gm_target_norm = get_smoothed_normal(gm_surface, gm_elm_index, smooth=0)

# Get direction of vector pointing from gm target to wm shortest-distance pair
pair_direction = (wm_pair_pos - gm_target_pos) / wm_pair_dist[wm_elm_index]

# Angular difference bewteen the two directions as a metric of agreement
angle_diff = np.arccos(np.clip(np.dot(-1*gm_target_norm, pair_direction), -1.0, 1.0)) * (180 / np.pi)

# Visualize the brain mesh, target region, and direction vectors
def vis_pv(meshes, opacities, center, directions, labels):
    plotter = Plotter()

    # Plot meshes
    cmaps = ['coolwarm', 'bwr']
    for m, o, c in zip(meshes, opacities, cmaps): plotter.add_mesh(m, opacity=o, cmap=c)
    
    # Plot direction vectors
    colors = ['#39FF14', '#FFFF00'] # Green, Yellow
    for d, col, label in zip(directions, colors, labels):
        plotter.add_arrows(np.array(center), np.array(d), mag=3, color=col, label=label)
    
    # Set focus to target
    plotter.set_focus(center)

    # Position camera to look at target from above
    cam_pos = [center[0], center[1], center[2]+50]
    plotter.set_position(cam_pos)
    plotter.camera.roll = 180
    
    plotter.add_legend()
    plotter.show()

print('Target position:' + str(target))
print('GM surface element closest to target: ' + str(gm_target_pos))
print('Distance between target coords & GM target: ' + str(np.linalg.norm(gm_target_pos - target)))
print('WM surface element closest to GM target: ' + str(wm_pair_pos))
print('Distance between GM target & WM pair point: ' + str(np.linalg.norm(gm_target_pos - wm_pair_pos)))
print('Norm direction of GM surface target (pointing into brain): ' + str(-1*gm_target_norm))
print('Displacement direction between GM target & WM pair point: ' + str(pair_direction))
print('Angle between norm & displacement directions (degrees): ' + str(angle_diff))
if vis_option:
    vis_pv([gm_pv_mesh, wm_pv_mesh], [0.5, 1], gm_target_pos, [-1*gm_target_norm, pair_direction], ['Norm', 'Displacement'])
