"""
DCA-YOLO Inference Script
=========================
Rice Leaf Disease Detection using DCA-YOLO

This code is associated with the manuscript submitted to The Visual Computer:
"Lightweight Visual Detection Framework for Real-Time Rice Leaf Disease
Identification on Edge Mobile Robots"

If you use this code, please cite our paper.
"""
# python tools/inference.py --weights runs/train/DCA-YOLO/weights/best.pt --source datasets/test/images/bacterial_blight_331.jpg
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# Disease class names
CLASS_NAMES = {
    0: "Rice Blast",
    1: "Brown Spot",
    2: "Bacterial Blight",
}


def parse_args():
    parser = argparse.ArgumentParser(description="DCA-YOLO Rice Disease Detection")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/DCA_YOLO.pt",
        help="Path to model weights file",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source: image file, directory, or 0 for webcam",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Directory to save detection results",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run on: cpu, 0, 0,1,2,3 (default: auto)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save result images",
    )
    return parser.parse_args()


def run_inference(args):
    # Load model
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input sources
    source = args.source
    if source.isdigit():
        source = int(source)  # webcam

    # Run inference
    print(f"Running inference on: {source}")
    print(f"Confidence threshold: {args.conf} | IoU threshold: {args.iou}\n")

    t_start = time.time()

    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    t_end = time.time()
    total_time = t_end - t_start
    num_images = len(results)

    # Process and display results
    for i, result in enumerate(results):
        img_path = result.path if result.path else f"frame_{i:04d}"
        img_name = Path(img_path).name

        boxes = result.boxes
        num_detections = len(boxes) if boxes is not None else 0

        print(f"[{i+1}/{num_images}] {img_name}  —  {num_detections} detection(s)")

        if boxes is not None and num_detections > 0:
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                print(
                    f"    {cls_name:20s}  conf={conf:.3f}  "
                    f"box=[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]"
                )

        # Save annotated image
        if not args.no_save:
            annotated = result.plot()
            save_path = output_dir / img_name
            cv2.imwrite(str(save_path), annotated)

    # Summary
    avg_ms = (total_time / num_images * 1000) if num_images > 0 else 0
    fps = num_images / total_time if total_time > 0 else 0
    print(f"\nProcessed {num_images} image(s) in {total_time:.2f}s")
    print(f"Average: {avg_ms:.1f} ms/image  |  {fps:.1f} FPS")
    if not args.no_save:
        print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)