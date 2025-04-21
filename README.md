# AlzhiNet-based model - CNN MRI Classifier for Early Classification of Alzheimer's Disease

This project uses a Convolutional Neural Network (CNN) and clinical database to classify Alzheimer's disease stages using `.jpg` MRI images.

## Project Structure
- `CNNTesting.py` - Main training and evaluation script.
- Data: Folders of MRI `.jpg` images segmented into dataset for traning, validating and testing.
- Labels: An Excel file with subject IDs and class labels (.csv databased pulled from NACCC)

## 🛠️ Requirements
- Python 3.9+
- TensorFlow / Keras
- NumPy, Pandas, OpenCV

## Batch installation
- Downloads the requirements.txt.
- Run pip install -r requirements.txt

Or, simply use an IDE and install the dependent modules.
Install with PyCharm: https://www.jetbrains.com/pycharm/download/?section=windows

## 🚀 How to Run
1. Prepare your dataset and labels.
2. Update the paths in `CNNTesting.py`:
   - Image directory
   - Excel file path
3. Run the script: py CNNTesting.py (or python3 for Mac users).

