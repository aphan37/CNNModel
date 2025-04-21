# AlzhiNet-based model - CNN MRI Classifier for Early Classification of Alzheimer's Disease

This project implements AlzhiNet, a custom convolutional neural network (CNN) for detecting Alzheimer’s stages from MRI images that probable to achieve ultra high accuracy (98.95% - 99.02%) (in `.jpg` format). The pipeline includes data preprocessing, training, evaluation, explainability (via Grad-CAM), and a user interface (CLI + Streamlit GUI).

> This research was supported through the SURI initiative under the mentorship of Dr. Sriram Srinivasan and Dr. Ruth Agada. Team members contributing on this project: Lawrence Miggins, Chibueze Oburuoh, Kevin Elias Mejia, Darryl Lomax Jr, Lauren Buriss.

---

## Project Structure
- `CNNTesting.py` - Main training and evaluation script.
- Data: Folders of MRI `.jpg` images segmented into dataset for traning, validating and testing.
- Labels: An Excel file with subject IDs and class labels (.csv databased pulled from NACCC)

## 🛠️ Requirements
- Python 3.9+
- TensorFlow / Keras
- NumPy, Pandas, OpenCV

Install with PyCharm: https://www.jetbrains.com/pycharm/download/?section=windows
Or, follow the set up below.

## 🚀 How to Run
1. Prepare your dataset and labels.
2. Update the paths in `CNN.py`:
   - Image directory
   - Excel file path
3. Run the script: py CNN.py (or python3 for Mac users).
---
## 🚀 Features
- Organizes and splits image data from CSV  
- Trains a CNN (`AlzhiNet`) on labeled MRI images  
- Validates and evaluates model performance  
- Saves best model automatically (`best_model.pth`)  
- Includes Grad-CAM for model interpretability  
- Includes CLI + Streamlit GUI for user interaction

---
## 🛠️ Setup Instructions
1. Clone the repository:
   git clone https://github.com/aphan37/CNNTesting
   cd CNNTesting
2. install dependencies:
   pip install -r requirements.txt
3. Place your MRI .jpg files and corresponding database file (.csv) in the root directory.
4. Run python CNN.py
   
## Run Command Line Classifier
python gradcamCLI.py --image test_images/Mild.jpg --gradcam
## Run Web App (Streamlit GUI)
streamlit run CNN.py
## Expect Output (Theory)
- Confusion matrix
- Final test accuracy
- Grad-CAM visualizations showing model attention
