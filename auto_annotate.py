import cv2
import os
from ultralytics import YOLO

# ==========================================
# 1. Directory Setup (Adjust paths as needed)
# ==========================================
INPUT_IMAGES_FOLDER = r'D:\Upgrade model Yolo11n-pose\ultralytics\KLYVERO_Dataset\images'
OUTPUT_LABELS_FOLDER = r'D:\Upgrade model Yolo11n-pose\ultralytics\KLYVERO_Dataset\labels'

# Ensure the output directory exists
os.makedirs(OUTPUT_LABELS_FOLDER, exist_ok=True)

# ==========================================
# 2. Load the Base Model
# ==========================================
# Using the Nano version as it is sufficient for initial auto-annotation
model = YOLO('yolo11n-pose.pt') 

# ==========================================
# 3. Process Images & Extract the 12 Keypoints
# ==========================================
# Fetch all valid images from the input directory
valid_extensions = ('.png', '.jpg', '.jpeg')
image_files = [f for f in os.listdir(INPUT_IMAGES_FOLDER) if f.lower().endswith(valid_extensions)]

print(f"Found {len(image_files)} images. Starting Auto-Annotation...")

for filename in image_files:
    img_path = os.path.join(INPUT_IMAGES_FOLDER, filename)
    img = cv2.imread(img_path)
    
    # Get image dimensions (crucial for normalizing coordinates to 0-1 range)
    img_height, img_width, _ = img.shape
    
    # Run inference
    results = model(img)
    
    txt_filename = os.path.splitext(filename)[0] + '.txt'
    txt_path = os.path.join(OUTPUT_LABELS_FOLDER, txt_filename)
    
    with open(txt_path, 'w') as f:
        for result in results:
            # Ensure a person and keypoints were detected in the image
            if result.boxes is None or result.keypoints is None:
                continue
                
            # Extract Bounding Box coordinates in Normalized format (xywhn)
            boxes = result.boxes.xywhn.cpu().numpy() 
            
            # Extract Keypoint coordinates (X, Y, Confidence)
            keypoints = result.keypoints.data.cpu().numpy()
            
            # Iterate over each detected person (ideally, one person in front of the mirror)
            for i in range(len(boxes)):
                box = boxes[i]
                class_id = 0  # Class ID for 'person' is always 0
                
                # Extract keypoints for this specific person
                kpts = keypoints[i]
                
                # *** THE CORE LOGIC: Ignore the first 5 keypoints (face) and keep the 12 body keypoints ***
                body_kpts = kpts[5:] 
                
                # Write Bounding Box data first
                line = f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
                
                # Write the 12 keypoints data
                for kpt in body_kpts:
                    kx, ky, kconf = kpt
                    
                    # Normalize pixel coordinates relative to image dimensions
                    nx = kx / img_width
                    ny = ky / img_height
                    
                    # Determine Visibility status based on YOLO formats
                    if nx == 0 and ny == 0:
                        vis = 0  # Missing / Not in frame
                    elif kconf > 0.5:
                        vis = 2  # Visible and confident
                    else:
                        vis = 1  # Occluded (e.g., under clothes) or low confidence
                        
                    line += f" {nx:.6f} {ny:.6f} {vis}"
                    
                # Save the formatted line to the text file
                f.write(line + '\n')

print("✅ Auto-Annotation Completed Successfully!")