# A Novel Triple Hybrid Model for Gastrointestinal Disease Detection

## Description
This repository contains the official implementation of the research paper titled "A novel triple hybrid model with channel attention for advanced Gastrointestinal Disease Detection in Endoscopy". The project introduces a hybrid architecture combining ResNet50, SENet, and Vision Transformer (ViT) to enhance the diagnostic accuracy of endoscopic images.

## Dataset Information
The models were trained and validated using two publicly available datasets:
1. **Kvasir Dataset-v2**: A multi-class dataset containing 8,000 images representing 8 different classes of GI tract findings.
2. **Kvasir-Capsule**: A large-scale dataset for capsule endoscopy.
The Kvasir Dataset-v2 (URL: https://datasets.simula.no/kvasir/) and the Kvasir-Capsule dataset (URL: https://datasets.simula.no/kvasir-capsule/) were utilized in this study. These datasets are curated by external clinical sources and are widely recognized for GI tract anomaly detection.
*Note: Users must download the datasets from their official sources as per their respective licenses.*

## Code Information
The repository includes the following Jupyter Notebooks:
* `kvasir_hybrid_vit_transformer.ipynb`: Core implementation of the proposed ResNet50+SENet+ViT hybrid model.
* `kvasir_hybrid_swin_transformer.ipynb`: Implementation using Swin Transformer backbones.
* `feature_selection.ipynb`: Pre-processing and feature reduction scripts.
* Baseline Models: `densenet_121.ipynb`, `googlenet.ipynb`, `inception-v3.ipynb`, `resnet50.ipynb`, `vgg16.ipynb`, `vgg19.ipynb`, and `mobilenet_v2.ipynb` for comparative analysis.

## Requirements
To reproduce the results, the following dependencies are required:
* Python 3.9+
* PyTorch 2.0+
* Torchvision
* Scikit-learn
* Pandas & NumPy
* Matplotlib & Seaborn

Installation:
`pip install torch torchvision scikit-learn pandas numpy matplotlib seaborn`

## Usage Instructions
1. Clone the repository.
2. Ensure the datasets are placed in the `/data` directory or update the `DATASET_PATH` variable in the notebooks.
3. Run `feature_selection.ipynb` for data preparation.
4. Execute the hybrid model notebooks to begin training and evaluation.

## Methodology
The methodology follows a specific pipeline: 
1. Data Augmentation (Static and Dynamic).
2. Feature extraction via ResNet50.
3. Channel-wise attention refinement using SENet.
4. Global context capture using Vision Transformers.
5. Multi-class classification with a Softmax output layer.

## Citations
If you use this work, please cite:
Doğan, V., Aydilek, H., & Erten, M. Y. (2026). A novel triple hybrid model with channel attention for advanced Gastrointestinal Disease Detection in Endoscopy.

## License
This project is licensed under the MIT License.
