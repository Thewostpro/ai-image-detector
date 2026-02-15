# 🚀 Quick Reference Guide

## Essential Commands

### 1. Initial Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Check setup
python check_setup.py
```

### 2. Dataset Setup
```bash
# Create folders
mkdir -p dataset/train/ai dataset/train/real
mkdir -p dataset/test/ai dataset/test/real

# Add your images to these folders
```

### 3. Training
```bash
# Train the model
python train.py

# Expected time: 30-60 minutes (CPU), 5-15 minutes (GPU)
```

### 4. Testing
```bash
# Evaluate on test set
python test.py
```

### 5. Web App
```bash
# Launch Streamlit app
streamlit run app.py

# Access at: http://localhost:8501
# Press Ctrl+C to stop
```

### 6. Git Commands
```bash
# Initialize repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: AI Image Detector"

# Add remote (replace USERNAME)
git remote add origin https://github.com/USERNAME/ai-image-detector.git

# Push to GitHub
git push -u origin main
```

---

## File Structure
```
ai-image-detector/
├── train.py                    # Training script
├── test.py                     # Testing script
├── app.py                      # Streamlit web app
├── check_setup.py              # Environment checker
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── .gitignore                  # Git ignore rules
│
├── dataset/                    # Your data
│   ├── train/
│   │   ├── ai/
│   │   └── real/
│   └── test/
│       ├── ai/
│       └── real/
│
├── outputs/                    # Generated files
│   ├── best_model.keras
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
└── venv/                       # Virtual environment
```

---

## Common Issues & Quick Fixes

### Issue: Module not found
```bash
pip install -r requirements.txt
```

### Issue: Dataset not found
```bash
mkdir -p dataset/train/ai dataset/train/real
mkdir -p dataset/test/ai dataset/test/real
```

### Issue: Model file too large for GitHub
```bash
# Use Git LFS
git lfs install
git lfs track "*.keras"
git add .gitattributes
```

### Issue: Port already in use (Streamlit)
```bash
streamlit run app.py --server.port 8502
```

### Issue: Out of memory during training
Edit train.py and change:
```python
'batch_size': 16,  # Reduce from 32 to 16 or 8
```

---

## Dataset Requirements

### Minimum (for learning)
- Training: 200 images per class (400 total)
- Testing: 50 images per class (100 total)

### Recommended (for good results)
- Training: 1,000 images per class (2,000 total)
- Testing: 200 images per class (400 total)

### Professional (for best results)
- Training: 5,000+ images per class (10,000+ total)
- Testing: 1,000+ images per class (2,000+ total)

---

## Performance Benchmarks

| Dataset Size | Training Time (CPU) | Expected Accuracy |
|--------------|---------------------|-------------------|
| 400 images   | 15-20 mins         | 70-80%           |
| 2,000 images | 30-45 mins         | 85-92%           |
| 10,000 images| 60-90 mins         | 92-96%           |

*GPU training is 5-10x faster*

---

## Evaluation Metrics Explained

- **Accuracy**: Overall correct predictions
- **Precision**: When model predicts "AI", how often is it correct?
- **Recall**: Of all actual AI images, how many did we catch?
- **F1-Score**: Balance between precision and recall

---

## Troubleshooting Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated (see `(venv)` in terminal)
- [ ] All packages installed
- [ ] Dataset folders created and populated
- [ ] Project files in correct location
- [ ] Enough disk space (5-10 GB)
- [ ] Enough RAM (8 GB minimum)

---

**Keep this guide handy for quick reference!**