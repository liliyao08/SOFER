# SOFER: Enhancing Blockchain Detection with Symbolic Explainability and Feature Reduction

## Introduction

Blockchain technology has revolutionized numerous sectors, but detecting illicit activities within decentralized systems remains a significant challenge. Existing detection models often struggle to balance accuracy and explainability, especially as illicit activities within blockchain environments evolve rapidly. To address this, we propose **SOFER** (SymbOlic Feature Explanation Reduction), a novel approach that integrates imitation learning with **Kolmogorov-Arnold Networks (KAN)** to create a highly precise and interpretable model. SOFER not only improves detection accuracy but also enhances explainability by generating multiple, mutually corroborating explanations. Additionally, it incorporates feature reduction techniques to improve computational efficiency, making it a flexible and scalable solution for blockchain regulatory needs. We demonstrate the effectiveness of SOFER through evaluations on benchmark datasets, showing significant improvements in both performance and explainability compared to existing models. The **SOFER** code is available at [this GitHub repository](https://github.com/liliyao08/SOFER).

## Getting Started

This project is based on Python 3.10 and requires the installation of specific dependencies.

### Prerequisites

To install the required dependencies, first install `pykan` by following the instructions from [pykan GitHub repository](https://github.com/KindXiaoming/pykan).

Then, install the other required libraries by using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Running SOFER

To run the **SOFER** model, execute the following command:

```bash
python SOFER.py
```

You can modify temporary settings by changing the following code in `SOFER.py`:

```python
X = pd.read_csv(f'./features/X_1_0.7086.csv.csv')  
y = pd.read_csv(f'./features/y_1_0.7086.csv.csv')  
X = X[0:50000]  
y = y[0:50000]
```

### Feature Selection

To perform feature selection on the dataset, run the following script:

```bash
python feature_selection.py
```

## Full Pipeline Execution

### Datasets

1. **BABD-13**: [Download from Kaggle](https://www.kaggle.com/datasets/lemonx/babd13)
    
2. Additional datasets are discussed in the paper.
    

### Feature Selection

To select features, run the `feature_selection.py` script.

### Running SOFER

Once feature selection is completed, you can run the **SOFER** model:

```bash
python SOFER.py
```

### Comparison Algorithms

To test the performance of the proposed algorithm under different sub-models, use the feature-selected data with the `Classic_models.py` script. To fully test other models, you should use the raw, unprocessed **BABD-13** dataset.

Run the classic comparison models using:

```bash
python Classic_models.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://sorryios.ai/c/LICENSE) file for details.

## Acknowledgments

- We thank the authors and contributors of the **BABD-13** dataset for making it publicly available.
    
- Thanks to the developers of **pykan** and other libraries used in this project.
    
