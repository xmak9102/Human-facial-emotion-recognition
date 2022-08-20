1. Get data: 
- Raw data (Images): - Fer 2013: https://www.kaggle.com/msambare/fer2013
            - CK+: https://www.kaggle.com/code/shawon10/ck-facial-expression-detection
- Pixels data (Feather format): ml_midterm_last (unzip in colab notebook)
- Features (Feather format): https://drive.google.com/drive/folders/1dyH8KUdNF18yYOq8wFYubKXCec2IOtp2
2. Run Colab notebooks for tradional ML methods and MLP  (Use feather format data processed)
- get data from drive 
- package: sklearn, dlibs, pytorch
- START HERE: Set up -> EDA, Feature engineering, tuning model with landmarks
- run the Apply full feature section for HOG + Landmark features 
3. Preprocess data + CNN models
- package: pytorchlightning, weight&bias, hydra, sklearn, pytorch, albumentations (augment)
- Get Raw data (can use download.sh)
- Preprocess: src/preprocess.py, processed.py
- Augmentation: src/augmentation.py -> 2 options full or 1 (uncomment your choice, default=1)
- Dataset: src/dataset.py
- Train or inference from checkpoint: (read the config in config folder)
    Hydra CLI: 
    Example: model resnet18, 20 epochs, learning_rate=1e-4, testing and validating: "10-crop", batch_size=64, optimizer="adam", tuning = True (not use validation set),  use pretrained Model, training=True (testing use it), resume_training="checkpoint", load_checkpoint="" (for testing and change training )
    !python3 src/models/train_with_pl.py train.training.model=resnet18 train.training.epochs=20 train.optim.lr=1e-4 test.method="10-crop" train.loader.batch_size=64 train.optim.optimizer=adam  train.training.tuning=True train.training.pretrained=True  train.training.train_mode=True train.training.resume_from_ckp=None train.training.load_from_ckp=None
- get pretrained checkpoints:   
- PL_tutorial for learning 
- get pretrained ckp: .zip
4. For example of usage: midterm_usage.ipynb
