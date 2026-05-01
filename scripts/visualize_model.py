import blenderproc as bproc
import numpy as np
import os
import sys
import argparse

def visualize_model(model_path, output_dir):
    bproc.init()
    
    # Load the model
    objs = bproc.loader.load_obj(model_path)
    if len(objs) > 1:
        objs[0].join_with_other_objects(objs[1:])
    obj = objs[0]
    
    # Center the object at origin
    bbox = obj.get_bound_box()
    center = np.mean(bbox, axis=0)
    obj.set_location(-center)
    
    # Get size for camera placement
    size = np.max(np.ptp(bbox, axis=0))
    dist = size * 2.5
    
    # Create axis indicators
    def create_axis(name, vec, color):
        axis = bproc.object.create_primitive('CYLINDER', scale=[size*0.01, size*0.01, size*0.5])
        axis.set_name(f"Axis_{name}")
        # Rotate cylinder to point along vec
        # Cylinder default is along Z
        if name == 'X':
            axis.set_rotation_euler([0, np.pi/2, 0])
        elif name == 'Y':
            axis.set_rotation_euler([np.pi/2, 0, 0])
        
        # Position so it starts at origin and points out
        axis.set_location(np.array(vec) * size * 0.25)
        
        mat = bproc.material.create(f"Mat_{name}")
        mat.set_principled_shader_value("Base Color", color)
        axis.replace_materials(mat)
        return axis

    create_axis('X', [1, 0, 0], [1, 0, 0, 1]) # Red
    create_axis('Y', [0, 1, 0], [0, 1, 0, 1]) # Green
    create_axis('Z', [0, 0, 1], [0, 0, 1, 1]) # Blue

    # Set up lighting
    light = bproc.types.Light()
    # Scale energy by size squared to account for inverse square law
    # 500 was okay for size 1, so for size S we need 500 * S^2
    light.set_energy(500 * (size**2))
    light.set_location([size*2, size*2, size*2])
    
    # Add ambient world background to prevent grey/black images
    bproc.renderer.set_world_background([0.2, 0.2, 0.2], strength=1.0)
    
    # Configure renderer for basic visualization
    bproc.renderer.set_max_amount_of_samples(16)
    try:
        bproc.renderer.set_denoiser('OPENIMAGEDENOISE')
    except:
        pass
    # Setup camera for 4 views
    import bpy
    bpy.context.scene.camera.data.clip_end = max(dist * 10, 5000.0)
    
    # 1. Perspective/ISO
    cam_pos_iso = [dist, dist, dist]
    rotation_matrix = bproc.camera.rotation_from_forward_vec(np.array([0,0,0]) - np.array(cam_pos_iso))
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_pos_iso, rotation_matrix))
    
    # 2. Front (XZ plane, looking at +Y)
    cam_pos_front = [0, -dist, 0]
    rotation_matrix = bproc.camera.rotation_from_forward_vec([0, 1, 0])
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_pos_front, rotation_matrix))
    
    # 3. Side (YZ plane, looking at +X)
    cam_pos_side = [-dist, 0, 0]
    rotation_matrix = bproc.camera.rotation_from_forward_vec([1, 0, 0])
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_pos_side, rotation_matrix))
    
    # 4. Top (XY plane, looking at -Z)
    cam_pos_top = [0, 0, dist]
    rotation_matrix = bproc.camera.rotation_from_forward_vec([0, 0, -1])
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam_pos_top, rotation_matrix))

    # Render
    data = bproc.renderer.render()
    
    # Save images
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(os.path.dirname(model_path))
    
    import imageio
    view_names = ["ISO", "Front", "Side", "Top"]
    for i, name in enumerate(view_names):
        out_path = os.path.join(output_dir, f"{model_name}_{name}.png")
        imageio.imwrite(out_path, data['colors'][i])
        print(f"Saved {name} view to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the .obj file")
    parser.add_argument("--output_dir", default="model_visualizations", help="Output directory")
    args = parser.parse_args()
    
    visualize_model(args.model_path, args.output_dir)
