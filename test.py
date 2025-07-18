import argparse
import os
import time
import numpy as np
from ultralytics import YOLO

def main(args):
    # Load model
    model = YOLO(args.model_path)

    # Get image paths
    image_paths = [os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    total_time = 0
    num_images = len(image_paths)
    print(f"\n=== Detect với threshold {args.conf:.2f} ===")

    for image_path in image_paths:
        start = time.time()
        results = model.predict(source=image_path, save=False, conf=args.conf, imgsz=args.imgsz, verbose=False)
        end = time.time()

        elapsed = end - start
        total_time += elapsed

        num_objects = len(results[0].boxes)
        print(f"{os.path.basename(image_path)}: {num_objects} objects detected in {elapsed:.3f} seconds")

    # Statistics
    if num_images > 0:
        avg_time = total_time / num_images
        print(f"\nProcessed {num_images} images.")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Average inference time per image: {avg_time:.3f} seconds")
    else:
        print("No images found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Custom Detection Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to custom YOLO model")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing test images")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")

    args = parser.parse_args()
    main(args)
