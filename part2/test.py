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
# Load the saved U-Net model with ResNet18 as backbone
def load_model(model_path, num_classes=1):
    model = UNetResNet18Scratch(num_classes=num_classes).to(device)
    # Use map_location to ensure it loads onto the correct device
    model.load_state_dict(torch.load(model_path, map_location=device))
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
        predicted_mask = torch.sigmoid(output).squeeze().numpy() #.cpu().numpy()  # Apply sigmoid and convert to NumPy array

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

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Path to input video and model
    video_path = '/home/deepak/PhD/PhD/course_work/rbe474x_p2/test_video.mp4'  # Replace with the path to your input video
    model_path = '/home/deepak/Downloads/p1.pth'  # Replace with the pth file downloaded from the report
    
    # Load the model
    model = load_model(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video writers for part2.mp4 (semantic segmentation)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_semantic = cv2.VideoWriter('output/part2.mp4', fourcc, fps, (width * 2, height))  # Concatenated: input + semantic segmentation

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break  # End of video

        # Perform semantic segmentation
        concatenated_semantic_frame, semantic_mask = perform_segmentation(frame, model)
        out_semantic.write(concatenated_semantic_frame)

    # Release resources
    cap.release()
    out_semantic.release()
    cv2.destroyAllWindows()

    print("Processing complete. Video saved as 'part2.mp4'.")
