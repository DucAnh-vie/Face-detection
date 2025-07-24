import argparse
import os
import time
from ultralytics import YOLO

def main(args):
    # Load model
    model = YOLO(args.model_path)

    # Warm-up (1 ảnh giả)
    model.predict(source=os.path.join(args.image_folder, os.listdir(args.image_folder)[0]),
                  imgsz=args.imgsz, conf=args.conf, verbose=False)

    # Bắt đầu đo thời gian batch
    print(f"\n=== Detect batch với threshold {args.conf:.2f} ===")
    start = time.time()

    # Predict nguyên folder
    results = model.predict(source=args.image_folder,
                            imgsz=args.imgsz,
                            conf=args.conf,
                            save=False,
                            verbose=False)

    end = time.time()

    # Thống kê
    total_time = end - start
    num_images = len(results)
    avg_time = total_time / num_images if num_images > 0 else 0

    print(f"Processed {num_images} images.")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average inference time per image: {avg_time:.3f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Batch Detection Script with Warm-up")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model .pt file")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing test images")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")

    args = parser.parse_args()
    main(args)
