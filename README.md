# PPE Detection Model

This repository contains code and models for detecting Personal Protective Equipment (PPE) using YOLOv8.

## Project Structure

- `dataset/`: Contains the dataset in YOLO format (images and labels)
- `runs/`: Contains training outputs (models, logs, results)
- `train_yolo*.py`: Training scripts for different model variants and epochs
- `detect_*.py`: Detection scripts (webcam, batch)
- `evaluate_*.py`: Evaluation scripts
- `FINAL_SUMMARY.md`: Summary of training results and recommendations

## Usage

### Training
To train a model, run one of the training scripts, e.g.:
```bash
python train_yolo_m_200epochs.py
```

### Evaluation
To evaluate the best model on the test set:
```bash
python evaluate_best.py
```

### Detection
To run detection on images:
```bash
python detect_ppe.py --images dataset/images/test --output runs/detect/ppe_detection_test
```

To run real-time webcam detection:
```bash
python detect_webcam_ppe.py
```

## Requirements

- Python 3.8+
- Ultralytics (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)
- Other dependencies as per the scripts

## Notes

- All models were trained on CPU due to lack of GPU availability.
- The dataset suffers from class imbalance, which affects performance on rare classes (especially "no_" classes).
- The target mAP50-95 of 0.90 was not achieved with the current dataset and training constraints. See `FINAL_SUMMARY.md` for details and recommendations.

## License

This project is for educational and demonstration purposes.
