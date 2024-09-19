import bpy
import random
import math

# Define ranges for random values
x_position_range = (-4, 7) #Ranges decided by the visibility limits based on camera and Objects positions
y_position_range = (-4, 10)
z_position_range = (-2, 2)
rotation_range = (-math.pi/2, math.pi/2)

# Get the objects by name 
obj1 = bpy.data.objects['Plane']
obj2 = bpy.data.objects['Plane.001']
cube = bpy.data.objects['Cube']
camera = bpy.data.objects['Camera']

# Set Object Index for each object to distinguish in the segmentation mask
obj1.pass_index = 1
obj2.pass_index = 2

view_layer = bpy.context.view_layer

view_layer.use_pass_object_index = True

# Set up the compositing nodes for mask generation
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
tree.nodes.clear()

# Add Render Layers node
render_layers = tree.nodes.new(type="CompositorNodeRLayers")

# Add ID Mask nodes for each object
id_mask1 = tree.nodes.new(type="CompositorNodeIDMask")
id_mask1.index = 1  # Object1 mask
id_mask2 = tree.nodes.new(type="CompositorNodeIDMask")
id_mask2.index = 2  # Object2 mask


# Add Mix node to combine both masks into one image
mix_node = tree.nodes.new(type="CompositorNodeMixRGB")
mix_node.blend_type = 'ADD'  # Add the two masks together

# Add file output node for saving the mask
file_output = tree.nodes.new(type="CompositorNodeOutputFile")
file_output.base_path = r'C:\Users\vshit\Desktop\rbe474x_p2\masks_close'  

# Add composite node for combining the mask output
composite = tree.nodes.new(type="CompositorNodeComposite")

# Connect the nodes
tree.links.new(render_layers.outputs["IndexOB"], id_mask1.inputs["ID value"])
tree.links.new(render_layers.outputs["IndexOB"], id_mask2.inputs["ID value"])

# Connect the ID masks to the Mix node
tree.links.new(id_mask1.outputs["Alpha"], mix_node.inputs[1])  # First input of Mix
tree.links.new(id_mask2.outputs["Alpha"], mix_node.inputs[2])  # Second input of Mix

# Output the combined mask
tree.links.new(mix_node.outputs["Image"], file_output.inputs[0])

# Connect the raw render (Image) output to the composite node to display the render result
tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])

# Set the number of iterations
iterations = 9
for i in range(iterations):
    
    # Generate random position and rotation for Object1
    random_x1 = random.uniform(-0.5,3)
    random_y1 = random.uniform(y_position_range[0], y_position_range[1])
    random_z1 = random.uniform(4.5,5.5)
    
    random_rot_x1 = random.uniform(rotation_range[1]/2, rotation_range[1])
    random_rot_y1 = random.uniform(rotation_range[0], rotation_range[1])
    random_rot_z1 = random.uniform(rotation_range[0]/3, rotation_range[1]/3)
    
    obj1.location = (random_x1, -29 , random_z1)
    obj1.rotation_euler = (random_rot_x1, random_rot_y1, random_rot_z1)
    
    # Generate random position and rotation for Object2
    random_x2 = random.uniform(-0.5,3)
    random_y2 = random.uniform(y_position_range[0], y_position_range[1])
    random_z2 = random.uniform(4.5,5.5)
    
    random_rot_x2 = random.uniform(rotation_range[1]/2, rotation_range[1])
    random_rot_y2 = random.uniform(rotation_range[0], rotation_range[1])
    random_rot_z2 = random.uniform(rotation_range[0]/3, rotation_range[1]/3)
    
    obj2.location = (random_x2, -26, random_z2)
    obj2.rotation_euler = (random_rot_x2, random_rot_y2, random_rot_z2)
    
    # Generate random position and rotation for Occlusion object
    random_cx = random.uniform(-0.5,3)
    random_cy = random.uniform(y_position_range[0], y_position_range[1])
    random_cz = random.uniform(4.5,5.5)
    
    random_rot_cx = random.uniform(rotation_range[0], rotation_range[1])
    random_rot_cy = random.uniform(rotation_range[0], rotation_range[1])
    random_rot_cz = random.uniform(rotation_range[0], rotation_range[1])
    
    cube.location = (random_cx, -27, 6)
    cube.rotation_euler = (random_rot_cx, random_rot_cy, random_rot_cz)
        
    # Set up file path and output settings for each iteration
    render_path = r'C:\Users\vshit\Desktop\rbe474x_p2\data_close'
    mask_path = r'C:\Users\vshit\Desktop\rbe474x_p2\masks_close'
    
    bpy.context.scene.render.filepath = f'{render_path}\\data_close{991+i}.png'
    file_output.file_slots[0].path = f'mask_close{991+i}.png'
    # Render the scene (image and mask)
    bpy.ops.render.render(write_still=True)
    
    # Print iteration information
    print(f"Iteration {i+1}: Rendered image and mask saved.")