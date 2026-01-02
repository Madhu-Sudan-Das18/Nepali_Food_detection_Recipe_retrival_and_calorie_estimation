# Food Recognition, Recipe Retrival and Calories Estimation - README
This is a Flask-based web application for food recognition using deep learning.
The app identifies food items from uploaded images & provides detailed recipes with nutritional info.
It includes user authentication, history tracking, and recipe scaling based on serving sizes.

## Features

### Core Functionality
- **Food Recognition**: Uses a pre-trained EfficientNet-B0 model to identify food items from uploaded images
- **Recipe Database**: Provides detailed recipes with ingredients and cooking instructions
- **Nutritional Information**: Calculates calorie counts based on ingredients
- **Serving Scaling**: Automatically adjusts ingredient quantities for different serving sizes

### User Management
- User registration and login with password hashing
- Session-based authentication
- Individual history tracking per user
- Default test user (demo/password123)

### Additional Features
- Multiple recipe categories and subcategories for each food item
- Top-3 prediction results with confidence scores
- Uploaded image storage and display
- Search history with timestamps
- Health information page

## Technical Stack

### Frontend
-**HTML,CSS,JS,Tailwind CSS**

### Backend
- **Framework**: Flask
- **Database**: SQLite3
- **Authentication**: Werkzeug security (password hashing)
- **Image Processing**: PIL/Pillow

### Machine Learning
- **Framework**: PyTorch
- **Model**: EfficientNet-B0 (fine-tuned)
- **Inference**: GPU/CPU compatible (automatic detection)
- **Preprocessing**: Standard ImageNet normalization

### Frontend
- HTML templates with Flask templating
- JavaScript for dynamic interactions
- AJAX for API calls

## Installation

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (optional, for GPU acceleration)

### Dependencies
Install required packages:
```bash
pip install flask torch torchvision timm pillow werkzeug

###File Structure 
project/
├── app.py                    # Main application
├── best_food_model.pth      # Trained model weights
├── recipes.json             # Recipe database
├── food_history.db          # SQLite database (auto-created)
├── templates/               # HTML templates
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   └── health_info.html
├── static/
│   └── uploads/            # Uploaded images
└── README.txt              # This file

### ML-Training-Script README 

This Jupyter notebook trains a deep learning model for multi-class image classification of Nepali and
Indian food dishes using PyTorch and the EfficientNet-B0 architecture from the TIMM library.

The script achieves approximately 94% validation accuracy and a weighted F1-score of 0.938 on a dataset with 46 classes.

## Features
- Uses GPU acceleration if available .
- Applies ImageNet-standard normalization and data augmentation (resize to 224x224, random horizontal flip for training).
- Trains for 20 epochs with Adam optimizer  and CrossEntropyLoss.
- Evaluates with accuracy, confusion matrix, and classification report using scikit-learn
- Saves trained model as `bestfoodmodel.pth` including class names

## Dataset
- Expects ImageFolder structure in `train_dir` and `val_dir` (paths set to `r'D:\...'` in notebook).[file:1]
- 49 classes of food items, primarily Nepali dishes like DalBhat, Momo, Selroti, Yomari, with some others (e.g., Pizza, Burger)
- Validation set: ~3689 samples; example class supports: Chiya (251), Selroti (279)

## Requirements
torch
torchvision
timm==1.0.21
matplotlib
scikit-learn
seaborn
numpy

Install via !pip install timm .

## Usage
1. Update `train_dir` and `val_dir` to point to your ImageFolder datasets.
2. Run all cells sequentially in Jupyter (installs deps, loads data, trains, evaluates).
3. Model saves to `bestfoodmodel.pth` after training; load with `torch.load` for inference.

## Training Results
| Epoch | Train Loss | Val Loss | Val Acc (%) |
|-------|------------|----------|-------------|
| 1     | 1.2801    | 0.5345  | 85.88      |
| 5     | 0.0424    | 0.2214  | 93.79      |
| 10    | 0.0223    | 0.2386  | 93.63      |
| 17    | 0.0103    | 0.2208  | 94.50      |
| 20    | 0.0136    | 0.2616  | 93.82      |

High F1-scores (>0.95) for common classes like Chiya (0.99), Momo (0.97); lower for rarer ones like MeatCurry (0.80)

## Model Inference Example
import torch
import timm

model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=46)
checkpoint = torch.load('bestfoodmodel.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
classes = checkpoint['classes']
# Predict on new image...
Adapt transforms to match training (resize 224x224, normalize [0.485,0.456,0.406]/[0.229,0.224,0.225]).


