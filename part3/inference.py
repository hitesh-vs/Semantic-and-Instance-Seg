import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
from network import UNetResNet18Scratch

# Set up the device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Load the saved U-Net model with ResNet18 as backbone
def load_model(model_path, num_classes=1):
    print('model path, ', model_path)
    model = UNetResNet18Scratch(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define the image transformation (same as used during training)
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to the same input size as training
    transforms.ToTensor()  # Convert the image to a tensor
])

# Semantic segmentation function
def perform_segmentation(frame, model):
    # Convert the frame (BGR from cv2) to RGB and PIL format
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original_size = img.size  # Store original size for resizing back

    # Preprocess the image
    input_tensor = img_transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)  # Model output (logits)
        predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()  # Apply sigmoid and convert to NumPy array

    # Threshold the predicted mask to create binary segmentation
    predicted_mask = (predicted_mask > 0.5).astype('float32')  # Binary mask (0 or 1)

    # Resize predicted mask back to original size
    predicted_mask_img = Image.fromarray(predicted_mask * 255).convert('L')  # Convert to PIL image
    predicted_mask_img = predicted_mask_img.resize(original_size, Image.NEAREST)  # Resize to original size
    predicted_mask = np.array(predicted_mask_img)  # Convert back to NumPy for further processing

    # Convert the predicted mask into a BGR image (grayscale)
    predicted_mask_bgr = cv2.cvtColor(np.array(predicted_mask_img), cv2.COLOR_GRAY2BGR)

    # Concatenate the original and the segmented mask
    concatenated_frame = cv2.hconcat([frame, predicted_mask_bgr])

    return concatenated_frame, predicted_mask

# Preprocessing to remove small outliers (noise)
def remove_small_errors(predicted_mask, min_size=500):
    predicted_mask = (predicted_mask * 255).astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(predicted_mask)
    for label in range(1, num_labels):  # Skip background (label 0)
        if np.sum(labels_im == label) < min_size:
            predicted_mask[labels_im == label] = 0
    return predicted_mask

# Custom Watershed Algorithm for instance segmentation
def apply_watershed(predicted_mask):
    # Step 1: Distance Transform
    dist_transform = cv2.distanceTransform(predicted_mask.astype(np.uint8), cv2.DIST_L2, 5)

    # Step 2: Create sure foreground by thresholding the distance transform
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Step 3: Sure background by dilating the predicted mask
    sure_bg = cv2.dilate(predicted_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)

    # Step 4: Identify unknown regions by subtracting sure foreground from sure background
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 5: Label connected components in sure foreground
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # Step 6: Add 1 to all labels so background is 1 and foreground starts at 2
    markers = markers + 1

    # Step 7: Mark unknown regions as 0
    markers[unknown == 255] = 0

    return markers

# Instance segmentation function using custom Watershed
def perform_instance_segmentation(frame, semantic_mask, original_size):
    # Convert the binary semantic mask for instance segmentation
    semantic_mask = semantic_mask / 255.0  # Normalize the binary mask

    # Remove small segmentation errors (outliers)
    processed_mask = remove_small_errors(semantic_mask)

    # Apply the custom watershed algorithm
    markers = apply_watershed(processed_mask)

    # Create an empty overlay image for coloring the segments
    color_mask = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

    # Define fixed colors for different instances
    fixed_colors = [
        [255, 255, 255],  # White for the first object
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [0, 255, 255],    # Cyan
        [255, 0, 255],    # Magenta
    ]

    # Assign colors based on the watershed markers
    for label in range(2, len(fixed_colors) + 2):  # Skip background (1)
        color_mask[markers == label] = fixed_colors[label - 2]  # Assign color from fixed list

    # Resize to original size
    instance_segmentation_img = Image.fromarray(color_mask).resize(original_size, Image.NEAREST)
    instance_segmentation_bgr = cv2.cvtColor(np.array(instance_segmentation_img), cv2.COLOR_RGB2BGR)

    return instance_segmentation_bgr

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Path to input video and model
    video_path = '/home/deepak/test/rbe474x_p2/test_video.mp4'  # Replace with the path to your input video
    model_path = '/home/deepak/Downloads/p1.pth'  # Replace with the pth file downloaded from the report
    
    # Load the model
    model = load_model(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video writers for part3.mp4 (instance segmentation)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_instance = cv2.VideoWriter('output/part3.mp4', fourcc, fps, (width, height))  # Only instance segmentation color masks

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break  # End of video

        # Perform semantic segmentation
        concatenated_semantic_frame, semantic_mask = perform_segmentation(frame, model)

        # Perform instance segmentation based on the semantic mask
        instance_segmentation_frame = perform_instance_segmentation(frame, semantic_mask, (width, height))
        out_instance.write(instance_segmentation_frame)

    # Release resources
    cap.release()
    out_instance.release()
    cv2.destroyAllWindows()

    print("Processing complete. Video saved as 'part3.mp4'.")
