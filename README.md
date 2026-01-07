# MSCLCMI
Integrating Multi-Scale Features with A Two-Stage Contrastive Learning for circRNA-miRNA Interaction Prediction.



## Environment Requirement

To run this project, make sure you have the following Python environment:

```bash
numpy==1.24.4
scipy==1.8.1
torch-cluster==1.6.1+pt20cu118
torch-geometric==2.5.3
torch-scatter==2.1.1+pt20cu118
torch-sparse==0.6.17+pt20cu118
torch-spline-conv==1.2.2+pt20cu118
```

---

## Model Structure

- **`datapro.py`**: This script handles the reading, preprocessing, and transformation of data into a format suitable for model training and evaluation.
- **`model.py`**: the core model proposed in the paper.
- **`train.py`**: completion of a 5-fold cross-validation experiment.
- **`clac_metric.py`**: Computes various performance metrics and evaluation indicators to assess the model's accuracy and effectiveness.

---

## Datasets

The `datasets/` folder contains the benchmark datasets used in the experiments. 

