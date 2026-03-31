# Folder Structure

```
Fight_Fire_with_Fire/
│
├── README.md                        # Project overview, usage instructions, and examples
├── FOLDER_STRUCTURE.md              # This file
├── requirements.txt                 # Python dependencies (PyTorch, ART, OpenCV, etc.)
├── sample.jpg                       # Example test image
├── sample2.jpg                      # Additional test image
│
├── FSRCNN_Combiner.py               # Train/test defensive patches for Faster R-CNN
├── YOLOv2_Combiner.py               # Train/test defensive patches for YOLOv2
├── YOLOv4_Combiner.py               # Train/test defensive patches for YOLOv4
├── YOLOR_Combiner.py                # Train/test defensive patches for YOLOR
├── YOLOv8_Combiner.py               # Train/test defensive patches for YOLOv8
│
├── ObjectDetector/                  # Detector wrappers used by the Combiner scripts
│   ├── fjn_fasterrcnn.py            # Faster R-CNN wrapper
│   ├── fjn_yolov2.py                # YOLOv2 wrapper
│   ├── fjn_yolov4.py                # YOLOv4 wrapper
│   ├── fjn_yolor.py                 # YOLOR wrapper
│   └── fjn_yolov8.py                # YOLOv8 wrapper
│
├── trained_dfpatches/               # Pre-trained defensive patches (ready to use)
│   ├── FSRCNN/
│   │   ├── canary.png               # Canary patch trained for Faster R-CNN
│   │   └── wd.png                   # Woodpecker patch trained for Faster R-CNN
│   ├── YOLOv2/
│   │   ├── canary.png
│   │   └── wd.png
│   ├── YOLOv4/
│   │   ├── canary.png
│   │   └── wd.png
│   ├── YOLOR/
│   │   ├── canary.png
│   │   └── wd.png
│   └── YOLOv8/
│       ├── canary.png
│       └── wd.png
│
├── InitImages/                      # Seed images used to initialize patch generation
│   ├── 8.jpg
│   ├── 17.jpg
│   ├── 18.jpg
│   ├── 19.jpg
│   ├── 20.jpg
│   ├── 22.jpg
│   └── 23.jpg
│
├── Data/                            # Training and evaluation datasets
│   └── traineval/
│       └── VOC07_YOLOv8/            # VOC 2007 dataset adapted for YOLOv8
│           └── train_120/           # 120-sample training split
│               ├── adversarial/     # Adversarially attacked images (120 JPEGs)
│               ├── benign/          # Clean images (120 JPEGs)
│               └── benign_label/    # YOLO-format annotation files for benign images
│
├── datasets/                        # Compressed datasets and dataset references
│   ├── README.md                    # Links to full datasets for all detectors
│   ├── VOC07_YOLOv8_train_120.7z    # Compressed 120-sample training set
│   └── YOLOv8.7z                    # Compressed YOLOv8 dataset
│
├── FJNTraining/                     # Training experiment outputs and checkpoints
│   └── canary_cc_fool_yolov8_2.0_10/
│       └── exp_VOC07_120_22_80_50/  # Saved results for this experiment configuration
│
├── assets/                          # Images used in the README and paper
│   ├── idea.jpg                     # Diagram illustrating the defensive patch concept
│   ├── effective_analysis_ca.jpg    # Effectiveness analysis — canary defense
│   ├── effective_analysis_wd.jpg    # Effectiveness analysis — woodpecker defense
│   ├── effective_analysis_cawd.png  # Effectiveness analysis — combined defense
│   ├── response-do-1.png            # Visualization of a successful defense response
│   ├── response-error-1.png         # Visualization of a failed defense response
│   └── response-error-2.png         # Additional failure case visualization
│
├── Response/                        # Reviewer response materials (IEEE S&P 2025)
│   ├── README.md                    # Overview of the review responses
│   ├── A-Q1.pdf … A-Q6.pdf         # Responses to Reviewer A questions
│   ├── B-Q5.pdf, B-Q8.pdf          # Responses to Reviewer B questions
│   ├── C-Q1.pdf, C-Q2.pdf, C-Q6.pdf# Responses to Reviewer C questions
│   ├── D-Q1.pdf, D-Q2.pdf          # Responses to Reviewer D questions
│   └── D-Q3/                        # Visual comparison samples for Reviewer D Q3
│       ├── readme.MD
│       ├── 000377_adv.jpg           # Adversarial example
│       ├── 000377_canary.jpg        # Same image with canary defense applied
│       ├── 000377_woodpecker.jpg    # Same image with woodpecker defense applied
│       └── …                        # Additional comparison triplets (001606, 003858, 006505)
│
└── ultralytics/                     # YOLOv8 framework (local copy for customization)
    ├── cfg/                         # Model and dataset YAML configurations
    │   ├── datasets/                # Dataset configs (COCO, VOC, etc.)
    │   ├── models/                  # Architecture configs (v3, v5, v6, v8, RT-DETR)
    │   └── trackers/                # Tracker configs (ByteTrack, BoT-SORT)
    ├── data/                        # Data loading and augmentation
    │   ├── augment.py               # Augmentation transforms
    │   ├── base.py                  # Base dataset class
    │   ├── build.py                 # DataLoader construction
    │   ├── dataset.py               # Dataset class definitions
    │   ├── loaders.py               # Stream / file loaders
    │   └── utils.py                 # Data utility functions
    ├── engine/                      # Core training and inference engine
    │   ├── model.py                 # YOLO base model class
    │   ├── trainer.py               # Training loop
    │   ├── validator.py             # Validation loop and metrics
    │   ├── predictor.py             # Inference / prediction
    │   ├── exporter.py              # Export to ONNX, TFLite, CoreML, etc.
    │   └── results.py               # Result post-processing and display
    ├── models/                      # Model architecture definitions
    │   ├── yolo/                    # YOLO family (detect, classify, segment, pose)
    │   ├── rtdetr/                  # Real-Time DETR
    │   ├── sam/                     # Segment Anything Model (SAM)
    │   ├── fastsam/                 # Fast SAM variant
    │   └── nas/                     # Neural Architecture Search models
    ├── nn/                          # Neural network building blocks
    │   ├── autobackend.py           # Auto backend selection (PyTorch, ONNX, TF, etc.)
    │   ├── tasks.py                 # Task-specific network heads
    │   └── modules/                 # Reusable layers (Conv, blocks, heads, transformers)
    ├── trackers/                    # Multi-object tracking
    │   ├── byte_tracker.py          # ByteTrack implementation
    │   ├── bot_sort.py              # BoT-SORT implementation
    │   └── utils/                   # Kalman filter, matching, GMC
    ├── hub/                         # Ultralytics Hub integration
    │   ├── auth.py                  # Authentication
    │   └── utils.py                 # Hub utilities
    └── utils/                       # General utilities
        ├── checks.py                # Environment / dependency checks
        ├── loss.py                  # Loss functions
        ├── metrics.py               # mAP, precision, recall, etc.
        ├── plotting.py              # Visualization helpers
        ├── torch_utils.py           # PyTorch helpers
        └── callbacks/               # Training callbacks (TensorBoard, W&B, MLflow, etc.)
```

## Key Concepts

| Term | Meaning |
|------|---------|
| **Canary patch (C)** | A defensive patch that mimics an adversarial patch to fool the attacker's detector |
| **Woodpecker patch (W)** | A defensive patch that neutralizes adversarial patches by preventing them from working |
| **Combiner script** | The main entry point for training or testing a defensive patch against a specific detector |
| **FJN** | Flexible Jittering Network — the underlying detection wrapper used during patch training |
| **VOC 2007** | PASCAL VOC 2007 object detection benchmark used for training and evaluation |
