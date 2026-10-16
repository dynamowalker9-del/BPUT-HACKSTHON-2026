

import argparse
import os
import sys
from pathlib import Path

import yaml
import cv2
import torch
from ultralytics import YOLO



DEFAULT_CONFIG = {
    "detect_weights": "runs/detect/runs/detect/yolo26n_ppe-5/weights/best.pt",
    "detect_specialist_weights": "runs/detect/ppe_nohelmet_nogloves_finetune/weights/best.pt",
    "detect_specialist_class": "no_gloves",
    "detect_source": "dataset/images/test",
    "detect_output": "runs/detect/ppe_detection",
    "conf_thresh": 0.15,
    "person_conf_boost": 0.2,
    "webcam_imgsz": 768,
    "iou_thresh": 0.45,
    "device": "auto",
    "view_img": False,
    "line_width": 3,
    "device": "auto",
}




def load_config(config_path="config.yaml"):

    config_file = Path(config_path)

    if not config_file.is_file():
        print(
            f"[INFO] Config file not found: {config_path}\n"
            "[INFO] Using built-in default settings."
        )
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        if not isinstance(config, dict):
            raise ValueError("Configuration must be a YAML dictionary.")

        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)

        return merged_config

    except (OSError, yaml.YAMLError, ValueError) as exc:
        sys.exit(f"[ERROR] Could not load configuration: {exc}")



def parse_args():

    parser = argparse.ArgumentParser(
        description="Real-time PPE detection using YOLO."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml."
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained YOLO .pt weights."
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Camera index, image, video, folder, or stream URL."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory in which detection results are saved."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold, from 0 to 1."
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image size."
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold for non-maximum suppression."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: auto, cpu, 0, or cuda:0."
    )

    parser.add_argument(
        "--view-img",
        dest="view_img",
        action="store_true",
        help="Display annotated inference results."
    )

    parser.add_argument(
        "--no-view-img",
        dest="view_img",
        action="store_false",
        help="Disable the inference display window."
    )

    parser.set_defaults(view_img=None)

    parser.add_argument(
        "--nosave",
        action="store_true",
        help="Do not save annotated results."
    )

    return parser.parse_args()



def normalize_source(source):
   

    if isinstance(source, int):
        return source

    if isinstance(source, str):
        source = source.strip()

        if source.isdigit():
            return int(source)

    return source


def resolve_project_path(value, project_root):
    
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def is_webcam_source(source):
    

    return isinstance(source, int)



def validate_settings(weights, conf, imgsz, iou):

    if not os.path.isfile(weights):
        sys.exit(
            f"\n[ERROR] Model weights not found:\n"
            f"        {weights}\n"
            
        )

    if not 0.0 <= conf <= 1.0:
        sys.exit("[ERROR] Confidence must be between 0 and 1.")

    if not 0.0 <= iou <= 1.0:
        sys.exit("[ERROR] IoU threshold must be between 0 and 1.")

    if imgsz <= 0:
        sys.exit("[ERROR] Image size must be greater than zero.")



def print_hardware_info(device):

    print("\n=== Hardware Information ===")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU             : Not available; CPU inference is possible.")

    if device == "auto":
        selected_device = 0 if torch.cuda.is_available() else "cpu"
    else:
        selected_device = device

    print(f"Selected device : {selected_device}")
    print("============================\n")

    return selected_device



def check_webcam(source):
    

    if not is_webcam_source(source):
        return

    print(f"[INFO] Checking camera index {source}...")

    # DirectShow is generally suitable for Windows webcams.
    # If it fails, retry with OpenCV's default backend.
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()

        sys.exit(
            f"\n[ERROR] Cannot open camera index {source}.\n"
            "Possible solutions:\n"
            "1. Close other applications using the camera.\n"
            "2. Check Windows camera permissions.\n"
            "3. Try --source 1 for another camera.\n"
            "4. Check that the camera is connected.\n"
        )

    ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        sys.exit(
            f"[ERROR] Camera {source} opened, "
            "but no video frame could be read."
        )

    print("[INFO] Camera is working correctly.")




def main():

    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path

    # Load configuration.
    cfg = load_config(config_path)

    # Command-line arguments override config.yaml.
    weights = resolve_project_path(
        args.weights or cfg.get("detect_weights", cfg.get("model_weights")),
        project_root,
    )
    specialist_weights = cfg.get("detect_specialist_weights")
    if specialist_weights:
        specialist_weights = resolve_project_path(specialist_weights, project_root)

    raw_source = (
        args.source
        if args.source is not None
        else cfg["detect_source"]
    )

    source = normalize_source(raw_source)
    if isinstance(source, str) and not source.startswith(("http://", "https://", "rtsp://")):
        source = resolve_project_path(source, project_root)

    output = resolve_project_path(args.output or cfg["detect_output"], project_root)

    conf = (
        args.conf
        if args.conf is not None
        else cfg["conf_thresh"]
    )

    imgsz = (
        args.imgsz
        if args.imgsz is not None
        else cfg["webcam_imgsz"]
    )

    iou = (
        args.iou
        if args.iou is not None
        else cfg["iou_thresh"]
    )

    device = args.device or cfg.get("device", "auto")
    view_img = (
        args.view_img
        if args.view_img is not None
        else cfg.get("view_img", False) or is_webcam_source(source)
    )

    # Validate settings and model path.
    validate_settings(weights, conf, imgsz, iou)
    if specialist_weights and not os.path.isfile(specialist_weights):
        sys.exit(f"\n[ERROR] Specialist weights not found:\n        {specialist_weights}")

    # Display hardware information.
    selected_device = print_hardware_info(device)

    # Print detection settings.
    print("=== PPE Detection Configuration ===")
    print(f"Weights       : {weights}")
    print(f"Source        : {source}")
    print(f"Output        : {output}")
    print(f"Confidence    : {conf}")
    print(f"Image size    : {imgsz}")
    print(f"IoU threshold : {iou}")
    print(f"Device        : {selected_device}")
    print(f"Display       : {view_img}")
    print(f"Save results  : {not args.nosave}")
    print("===================================\n")
    check_webcam(source)
    print("[INFO] Loading PPE detection model...")
    try:
        model = YOLO(weights)
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to load YOLO model: {exc}")

    specialist_model = None
    specialist_class_name = str(cfg.get("detect_specialist_class", "no_gloves"))
    base_specialist_class_id = next(
        (idx for idx, name in model.names.items() if str(name).casefold() == specialist_class_name.casefold()),
        None,
    )
    specialist_class_id = base_specialist_class_id
    if specialist_weights:
        try:
            specialist_model = YOLO(specialist_weights)
        except Exception as exc:
            sys.exit(f"[ERROR] Failed to load specialist model: {exc}")
        specialist_class_id = next(
            (
                idx for idx, name in specialist_model.names.items()
                if str(name).casefold() == specialist_class_name.casefold()
            ),
            None,
        )
        if specialist_class_id is None:
            sys.exit(f"[ERROR] Specialist class '{specialist_class_name}' not found in specialist model.")
        if base_specialist_class_id is None:
            sys.exit(f"[ERROR] Specialist class '{specialist_class_name}' not found in primary model.")

    person_class_id = next(
        (idx for idx, name in model.names.items() if str(name).casefold() in {"person", "human"}),
        None,
    )
    person_conf = min(1.0, conf + float(cfg.get("person_conf_boost", 0.2)))

    def apply_class_thresholds_and_specialist(predictor):
        """Keep the base model for other classes and use the specialist for its target."""
        for result in predictor.results:
            boxes = (
                result.boxes.data
                if result.boxes is not None
                else torch.empty((0, 6), device=predictor.device)
            )

            if specialist_model is not None:
                base_boxes = boxes[boxes[:, 5].long() != base_specialist_class_id]
                specialist_result = specialist_model.predict(
                    source=result.orig_img,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=selected_device,
                    verbose=False,
                )[0]
                specialist_boxes = (
                    specialist_result.boxes.data
                    if specialist_result.boxes is not None
                    else boxes.new_empty((0, 6))
                )
                specialist_boxes = specialist_boxes[
                    specialist_boxes[:, 5].long() == specialist_class_id
                ]
                specialist_boxes = specialist_boxes.clone()
                specialist_boxes[:, 5] = base_specialist_class_id
                boxes = torch.cat((base_boxes, specialist_boxes), dim=0)

            if boxes.numel():
                class_ids = boxes[:, 5].long()
                thresholds = boxes.new_full((len(boxes),), float(conf))
                if person_class_id is not None:
                    thresholds[class_ids == person_class_id] = person_conf
                boxes = boxes[boxes[:, 4] >= thresholds]

            result.update(boxes=boxes)

    model.add_callback("on_predict_postprocess_end", apply_class_thresholds_and_specialist)

    print("[INFO] Model loaded successfully.")
    print(f"[INFO] Person confidence threshold: {person_conf:.2f}")
    if specialist_weights:
        print(f"[INFO] {specialist_class_name} specialist: {specialist_weights}")

    print("\n[INFO] Starting detection...")
    print("[INFO] Press Q in the detection window to exit, if supported.\n")

    try:
        # Run inference.
        results = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=selected_device,
            show=view_img,
            save=not args.nosave,
            project=output,
            name="results",
            exist_ok=True,
            stream=True,
            line_width=cfg.get("line_width", 3),
            verbose=False,
        )

        result_count = 0

        # Iterate over results so inference progresses frame by frame.
        for result in results:

            result_count += 1

            # Print detection summary.
            boxes = result.boxes
            detection_count = len(boxes) if boxes is not None else 0

            if detection_count > 0:
                detected_classes = []

                for box in boxes:
                    class_id = int(box.cls[0].item())
                    class_name = model.names.get(
                        class_id,
                        str(class_id)
                    )

                    detected_classes.append(class_name)

                print(
                    f"[DETECTION] Frame {result_count}: "
                    f"{detection_count} object(s) | "
                    f"{', '.join(detected_classes)}"
                )

        print("\n=== Detection Finished ===")
        print(f"Frames/images processed: {result_count}")

        if not args.nosave:
            print(f"Results directory: {os.path.abspath(output)}")

        print("==========================")

    except KeyboardInterrupt:
        print("\n[INFO] Detection stopped by user.")

    except Exception as exc:
        print(f"\n[ERROR] Detection failed: {exc}")
        raise

    finally:
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

