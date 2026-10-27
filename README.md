# PCSDiff: Diffusion-Based Bias Correction and Super Resolution Toward Practical Operational Medium-Term Precipitation Forecast

This repository contains the code used for "PCSDiff: Diffusion-Based Bias Correction and Super Resolution Toward Practical Operational Medium-Term Precipitation Forecast" ([https://arxiv.org/abs/2510.17852](https://arxiv.org/pdf/2609.06942))

## 📖 Introduction
Medium‑range precipitation forecasts are impaired by persistent systematic biases, lead‑time‑dependent error accumulation, and coarse spatial resolution, restricting their reliability for flood‑drought risk assessment. Existing AI correction techniques lack dedicated modeling for multi‑day dynamic bias evolution and proper meteorological constraints, often generating over‑smoothed rainfall structures, and cannot meet operational deployment demands. This work introduces PCSDiff, a cascaded task‑decoupled diffusion framework targeting 10‑day precipitation bias correction and downscaling. To jointly counteract temporal error drifts and reconstruct physically plausible local precipitation details, PCSDiff integrates the Precipitation Intensity‑aware Multi‑branch Decoder (PIMD) module for dynamic multi‑day error mitigation using synoptic‑temporal features, followed by a two‑phase conditional diffusion super‑resolution module to restore fine‑scale precipitation patterns. Evaluated against CMA‑CRA observations over China after global‑data training, PCSDiff cuts RMSE by 16.1% and lifts ACC by 13.9% relative to raw ECMWF forecasts at 3–10‑day lead times, and consistently outperforms mainstream deep‑learning baselines on both general and extreme‑precipitation metrics. Benefiting from a streaming inference pipeline, our method achieves low‑latency rolling forecasting for practical meteorological operations.

[figure1.pdf](https://github.com/user-attachments/files/34016369/figure1.pdf)




### Prerequisites
- Python 3.8+
- PyTorch 1.12+


## 🏃‍♂️ Training and Evaluation
1. Training
```bash
python train.py
```
2. Evaluation
```bash
python test.py
```
## 📜 Citation
waiting...

## 📧 Contact
For any questions or suggestions, please contact [Yuze Sun] at [syz23@mails.tsinghua.edu.cn] or open an issue on GitHub.

## 📄 License
This project is licensed under the MIT License.
