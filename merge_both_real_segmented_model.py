import os
import csv
from datetime import datetime
# from model_real_segmentation import load_model, segment_roof_binary
import cv2
from PIL import Image
from PIL.ExifTags import TAGS
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
import cv2

def load_model(model_path, num_classes):
    # Create a Mask R-CNN model with the same configuration used during training
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

def segment_roof_binary(model, image_path, confidence_threshold=0.5):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    input_image = F.to_tensor(image_rgb)
    
    # Run inference
    with torch.no_grad():
        prediction = model([input_image])
    
    # Process predictions
    masks = prediction[0]['masks']
    scores = prediction[0]['scores']
    
    # Filter predictions based on confidence threshold
    mask_indices = torch.where(scores > confidence_threshold)[0]
    
    # Create a segmented image (same size as input image)
    segmented_image = image.copy()
    
    for idx in mask_indices:
        # Get the mask and convert to numpy
        mask = masks[idx, 0].numpy()
        
        # Create binary mask (threshold at 0.5)
        current_mask = (mask > 0.5).astype(np.uint8)
        
        # Apply the mask to the original image
        segmented_image[current_mask == 1] = [0, 255, 0]  # Green color for segmented roof
    
    # Create a binary mask (same size as input image)
    binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    for idx in mask_indices:
        # Get the mask and convert to numpy
        mask = masks[idx, 0].numpy()
        
        # Create binary mask (threshold at 0.5)
        current_mask = (mask > 0.5).astype(np.uint8)
        
        # Combine masks (logical OR)
        binary_mask = np.logical_or(binary_mask, current_mask).astype(np.uint8)
    
    # Convert binary mask to 0-255 range for saving
    binary_mask_255 = binary_mask * 255
    
    # Calculate the number of white pixels
    white_pixel_count = np.sum(binary_mask_255 == 255)
    
    # Calculate percentage of white pixels
    total_pixels = binary_mask_255.size
    white_pixel_percentage = (white_pixel_count / total_pixels) * 100
    
    return {
        'binary_mask': binary_mask_255,
        'segmented_image': segmented_image,
        'white_pixel_count': white_pixel_count,
        'white_pixel_percentage': white_pixel_percentage,
        'total_image_pixels': total_pixels
    }

def get_image_timestamp(image_path):
    """
    Extracts the timestamp from image metadata.
    If no timestamp is found, returns 'Unknown'.
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data is not None:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "DateTime":
                        return value  # Format: "YYYY:MM:DD HH:MM:SS"
    except Exception as e:
        print(f"Error reading timestamp for {image_path}: {e}")
    return "Unknown"

def process_images_and_save_outputs(model_path, input_folder, output_folder, csv_file_path, num_classes, confidence_threshold=0.5):
    # Load the model
    model = load_model(model_path, num_classes)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    binary_mask_folder = os.path.join(output_folder, "binary_masks")
    segmented_image_folder = os.path.join(output_folder, "segmented_images")
    os.makedirs(binary_mask_folder, exist_ok=True)
    os.makedirs(segmented_image_folder, exist_ok=True)

    # Open CSV file to write the results
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(["Image Name", "White Pixel Count", "White Pixel Percentage", "Click Time"])

        # Process each image in the input folder
        for image_name in os.listdir(input_folder):
            if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(input_folder, image_name)

                # Extract the timestamp
                timestamp = get_image_timestamp(image_path)

                # Perform roof segmentation
                result = segment_roof_binary(model, image_path, confidence_threshold)

                # Save the binary mask
                binary_mask_path = os.path.join(binary_mask_folder, f"mask_{image_name}")
                cv2.imwrite(binary_mask_path, result['binary_mask'])

                # Save the segmented image
                segmented_image_path = os.path.join(segmented_image_folder, image_name)
                cv2.imwrite(segmented_image_path, result['segmented_image'])

                # Write pixel data and timestamp to the CSV file
                csv_writer.writerow([
                    image_name,
                    result['white_pixel_count'],
                    f"{result['white_pixel_percentage']:.2f}",
                    timestamp
                ])

                # Print status
                print(f"Processed: {image_name}")
        
        print(f"Results saved to CSV file: {csv_file_path}")
        print(f"Binary masks saved in: {binary_mask_folder}")
        print(f"Segmented images saved in: {segmented_image_folder}")

# Main function to execute the script
if __name__ == "__main__":
    model_path = r'C:\Users\harsh\OneDrive\Desktop\safar\folder4_rcnn_weights.pth'
    input_folder = r'C:\Users\harsh\OneDrive\Desktop\safar\imaes'
    output_folder = r'C:\Users\harsh\OneDrive\Desktop\safar\safar_pandeye'
    csv_file_path = r'C:\Users\harsh\OneDrive\Desktop\safar\safar.csv'
    num_classes = 2

    process_images_and_save_outputs(model_path, input_folder, output_folder, csv_file_path, num_classes)