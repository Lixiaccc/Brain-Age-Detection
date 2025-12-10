
# Brain Age Detection from Handwriting and Drawing

## Project Overview

Brain age prediction from neuroimaging has emerged as a biomarker for neurological health, but MRI-based approaches remain expensive and inaccessible for widespread screening. This project investigates whether age classification from handwriting and drawing can serve as a behavioral proxy for brain age assessment. We developed and compared multiple deep learning architectures (ResNet-18, ResNet-50, Vision Transformer, and a feature extraction-based MLP) to predict age across six classes spanning preschool (3-6 years) to seniors (65+ years).

## Project Structure

Brain-Age-Detection/
├── README.md
├── Compare&Plots.ipynb          # Model comparison visualization
├── Models/
│   ├── Feature_extraction.ipynb  # Extract handcrafted motor control features
│   ├── feature+ViT.ipynb        # Train Feature MLP and Vision Transformer
│   ├── resnet18_train.ipynb     # Train and fine-tune ResNet-18
│   ├── resnet50_train.ipynb     # Train and fine-tune ResNet-50
│   ├── test_Feature_MLP.ipynb   # Complete pipeline: feature extraction + inference


## File Descriptions

### Compare&Plots.ipynb
Generates comparative visualizations across all four model architectures, including:
- Accuracy, Macro-F1, and Macro-AUC bar charts
- Confusion matrices
- Performance analysis

### Models/Feature_extraction.ipynb
Extracts 68 handcrafted features from preprocessed handwriting and drawing images, including:
- Intensity statistics (mean, std, histogram bins)
- Laplacian variance (tremor detection)
- Stroke geometry (curvature, width, orientation)
- Skeleton topology (structural complexity)
- Motion dynamics (jerk, shakiness)

### Models/feature+ViT.ipynb
Training pipeline for:
1. **Feature MLP**: Operates on 68 handcrafted features expanded to 2,346 polynomial interaction terms
2. **Vision Transformer (ViT)**: Fine-tunes pre-trained ViT on handwriting/drawing images

### Models/resnet18_train.ipynb
Training and fine-tuning pipeline for ResNet-18 architecture on age classification task.

### Models/resnet50_train.ipynb
Training and fine-tuning pipeline for ResNet-50 architecture on age classification task.

### Models/test_Feature_MLP.ipynb
End-to-end inference pipeline:
1. Load and preprocess images
2. Extract features
3. Load trained Feature MLP model
4. Generate predictions

## Key Results

- **Feature MLP**: 86.3% accuracy, 87.3% macro-F1, 98.2% macro-AUC
- **ResNet-50**: 83.1% accuracy, 83.9% macro-F1, 97.1% macro-AUC
- **ResNet-18**: 79.4% accuracy, 80.7% macro-F1, 96.9% macro-AUC
- **Vision Transformer**: 73.1% accuracy, 71.0% macro-F1, 94.8% macro-AUC

## Clinical Applications

The trained models were applied to clinical populations, revealing motor age gaps:
- **Parkinson's Disease**: Young adults (18-40) predicted as 20-40 years older
- **Autism Spectrum Disorder + Conduct Disorder**: Children (7-12) predicted as 4-6 years younger

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- scikit-learn
- NumPy
- Pandas
- Matplotlib

## Usage

1. **Feature Extraction**: Run `Feature_extraction.ipynb` to extract features from your dataset
2. **Model Training**: Execute desired training notebook (`feature+ViT.ipynb`, `resnet18_train.ipynb`, or `resnet50_train.ipynb`)
3. **Inference**: Use `test_Feature_MLP.ipynb` for end-to-end prediction on new samples
4. **Visualization**: Run `Compare&Plots.ipynb` to generate performance comparisons
 
## Raw data, Processed data and the Trained Weight access link for LionMail only: 

https://drive.google.com/drive/folders/1M0vPJx27MuruReelKH6-2_F8bSdlq6Ty?usp=sharing

### Dataset/Best Models/
Contains saved model weights achieving the highest validation performance:
- **best_mlp_poly_features.pt**: Feature MLP (86.3% accuracy)
- **age_classifier_resnet50.pth**: ResNet-50 (83.1% accuracy)
- **age_classifier_resnet18.pth**: ResNet-18 (79.4% accuracy)
- **best_vit_age_classifier.pt**: Vision Transformer (73.1% accuracy)

Best Models/
│       ├── best_mlp_poly_features.pt      # Best Feature MLP weights
│       ├── best_vit_age_classifier.pt     # Best ViT weights
│       ├── age_classifier_resnet18.pth    # Best ResNet-18 weights
│       └── age_classifier_resnet50.pth    # Best ResNet-50 weights


