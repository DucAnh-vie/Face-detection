import argparse
import os
import time
import torch
from ultralytics import YOLO

def main(args):
    # Load YOLO model
    model = YOLO(args.model_path)

    # Warm-up using a dummy tensor instead of a real image
    dummy = torch.zeros((1, 3, args.imgsz, args.imgsz))
    model(dummy.to(args.device))  # Forward pass for warm-up

    print(f"\n=== Batch detection with threshold {args.conf:.2f} on device {args.device} ===")

    # Overall time measurement
    start = time.time()

    # Variables for detailed timing
    total_preprocess = 0
    total_inference = 0
    total_postprocess = 0
    count = 0

    # Predict with stream=True to avoid accumulating results in RAM
    results = model.predict(
        source=args.image_folder,
        imgsz=args.imgsz,
        conf=args.conf,
        save=False,
        stream=True,       # Generator mode to prevent memory accumulation
        device=args.device,
        verbose=False,
        batch=args.batch,
        max_det=100        # Limit maximum detections per image
    )

    # Iterate through results to count and accumulate timing
    for r in results:
        count += 1
        total_preprocess += r.speed['preprocess']
        total_inference += r.speed['inference']
        total_postprocess += r.speed['postprocess']

    end = time.time()

    # Calculate overall stats
    total_time = end - start
    avg_time = total_time / count if count > 0 else 0

    # Calculate average breakdown per image
    avg_pre = total_preprocess / count
    avg_inf = total_inference / count
    avg_post = total_postprocess / count

    # Print results
    print(f"Processed {count} images.")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average inference time per image: {avg_time:.3f} seconds")
    print(f"Speed: {avg_pre:.1f}ms preprocess, {avg_inf:.1f}ms inference, {avg_post:.1f}ms postprocess "
          f"per image at shape (1, 3, {args.imgsz}, {args.imgsz})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized YOLO Batch Detection Script with Speed Breakdown")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model .pt file")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing test images")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0=GPU, 'cpu'=CPU)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for inference")

    args = parser.parse_args()
    main(args)
