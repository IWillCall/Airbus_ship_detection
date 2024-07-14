import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def process_image(image_path, model, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  

    mask = model.predict(img)[0]
    
    mask = (mask > 0.2).astype(np.uint8) * 255
    
    return mask

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'Input')
    output_dir = os.path.join(script_dir, 'Output')
    model_path = os.path.join(script_dir, 'unet_model.h5')

    if not os.path.exists(input_dir):
        print(f"Error: Directory 'Model_Input' was not found in {script_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loading the model
    model = load_model(model_path)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Image processing
    for image_file in image_files:
        print(f"Processing {image_file}...")
        image_path = os.path.join(input_dir, image_file)
        mask = process_image(image_path, model)
        
        # Saving the mask
        mask_filename = os.path.join(output_dir, f"mask_{image_file}")
        cv2.imwrite(mask_filename, mask)

    print("Processing is complete. Masks are saved in the 'Model_Output' directory")

if __name__ == "__main__":
    main()