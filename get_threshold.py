import argparse
import numpy as np
from ultralytics import YOLO

def main(args):
    model = YOLO(args.model_path)
    conf_thresholds = np.arange(args.start_conf, args.end_conf + args.step, args.step)
    map_results = []

    print("=== Tìm threshold tốt nhất theo mAP50-95 ===")
    for conf in conf_thresholds:
        metrics = model.val(data=args.data_yaml, conf=conf, iou=args.iou, split=args.split, verbose=False)
        map_score = metrics.box.map
        map_results.append((conf, map_score))
        print(f"conf={conf:.2f} -> mAP50-95={map_score:.4f}")

    best_conf, best_map = max(map_results, key=lambda x: x[1])
    print(f"\nThreshold tốt nhất: {best_conf:.2f} (mAP50-95={best_map:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO mAP Threshold Finder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to YOLO model .pt file")
    parser.add_argument("--data_yaml", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--start_conf", type=float, default=0.1, help="Start confidence threshold")
    parser.add_argument("--end_conf", type=float, default=0.9, help="End confidence threshold")
    parser.add_argument("--step", type=float, default=0.1, help="Confidence threshold step size")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to validate on")

    args = parser.parse_args()
    main(args)
