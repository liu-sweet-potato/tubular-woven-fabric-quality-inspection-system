# tubular-woven-fabric-quality-inspection-system
![](./assets/architecture/Architecture.png)
An EfficientAD-based industrial vision inspection system for circular looms, designed to automatically detect and localize warp and weft anomalies in woven tubular fabrics. The system performs real-time defect inspection, identifying broken threads, skipped wefts, dirt contamination, and other production quality issues with high accuracy and low latency.
## Directory
```commandline
~\TUBULAR-WOVEN-FABRIC-QUALITY-INSPECTION-SYSTEM
├─defect_flow_image.py                              # tubular woven fabric quality inspection pipeline
├─gradio_image.py                                   # Gradio entry point
├─requirements.txt
├─README.md
│
├─assets                                            # datasets and model weights
│  ├─datasets
│  ├─demo
│  │    └─0-1.jpg                                   # sample test image
│  │
│  └─weights                                        # trained model weights
│      └─ckptSmall
│          ├─best_teacher.pth
│          ├─zhongyu_autoencoder_last.pth
│          ├─zhongyu_quantiles_last.npy
│          └─zhongyu_student_last.pth
│
└─dev
   ├─distillation_training.py                        # teacher model distillation
   ├─eval.py                                         # evaluation
   ├─models.py                                       # model definitions
   ├─train_reduced_student.py                        # student model training
   │ 
   ├─configs
   │   └─mvtec_train.yaml                            # student model training configuration
   │
   └─data_process
       └─data_loader.py                              # data loading
```
## Usage
```commandline
conda create -n zy python=3.10 -y
conda activate zy
pip install -r requirements.txt
python gradio_image.py
```

## Reference
https://github.com/rximg/EfficientAD/tree/main