import argparse
import os
import time
from ultralytics import YOLO

def main(args):
    # Load model
    model = YOLO(args.model_path)

    # Warm-up (1 ảnh giả)
    first_image = os.path.join(args.image_folder, os.listdir(args.image_folder)[0])
    model.predict(source=first_image, imgsz=args.imgsz, conf=args.conf, verbose=False)

    print(f"\n=== Detect batch với threshold {args.conf:.2f} ===")
    start = time.time()

    # Predict với stream=True để tránh tích tụ RAM
    results = model.predict(
        source=args.image_folder,
        imgsz=args.imgsz,
        conf=args.conf,
        save=False,
        stream=True,       # tránh dồn kết quả vào RAM
        verbose=False
    )

    # Duyệt qua từng ảnh để đếm số lượng kết quả
    count = 0
    for _ in results:
        count += 1

    end = time.time()

    # Thống kê
    total_time = end - start
    avg_time = total_time / count if count > 0 else 0

    print(f"Processed {count} images.")
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
