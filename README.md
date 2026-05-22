# ERIC-BuildSys2024

This repository contains the code for the paper: [ERIC: Estimating Rainfall with Commodity Doorbell Camera for Precision Residential Irrigation](https://dl.acm.org/doi/10.1145/3671127.3698186), BuildSys'24 **Best Paper Award**.

## Repository overview

This codebase includes two complementary pipelines for rain sensing:

- **ERIC-edge**: Traditional ML workflows that fuse visual and audio features, including ROI extraction, feature aggregation, rain detection (classification), and rainfall estimation (regression).

- **ERIC-cloud**: ResNet-based pipeline for image-based rain detection and rainfall estimation, including data slicing, label generation, training, evaluation, and visualization.

## Project structure

- ERIC-edge: Edge-side feature extraction and ML models with visual + audio inputs.
- ERIC-cloud: Cloud-side ResNet workflows for classification and regression on image data.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{liu2024eric,
  title={{ERIC}: Estimating Rainfall with Commodity Doorbell Camera for Precision Residential Irrigation},
  author={Liu, Tian and Jin, Liuyi and Stoleru, Radu and Haroon, Amran and Swanson, Charles and Feng, Kexin},
  booktitle={Proceedings of the 11th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys)},
  year={2024}
}

@article{liu2026robust,
  title={Robust Rainfall Estimation with Multimodal Sensing for Precision Residential Irrigation},
  author={Liu, Tian and Jin, Liuyi and Stoleru, Radu and Haroon, Amran and Swanson, Charles and Feng, Kexin},
  journal={ACM Transactions on Sensor Networks},
  volume={22},
  number={4},
  articleno={29},
  pages={1--25},
  year={2026},
  doi={10.1145/3734526}
}
```