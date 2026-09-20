# Wattson: IoT Power Side-Channel Intrusion Detection

Code accompanying the manuscript submitted to the *Journal of Network and
Computer Applications*. This repository detects DoS, port-scan, and
SSH brute-force attacks on a Raspberry Pi 3B by analyzing its AC power
draw, using a low-cost sensor (ACS712-05B current sensor + MCP3008
ADC, ~$4 total) instead of a laboratory-grade power analyzer.

Results are compared against Lightbody et al.'s prior work (Future
Internet, 2023/2024), which used a ~$6,500 laboratory power analyzer;
several scripts in this repository (`dragon_slice_*.py`) replicate
their unsupervised convolutional autoencoder as a baseline — "Dragon_Slice"
there refers to their published method name, not to this project.

## Hardware

- Raspberry Pi 3 Model B (target device)
- ACS712-05B current sensor (Allegro MicroSystems)
- MCP3008 10-bit SPI ADC (Microchip Technology)

## Repository structure

| Script | Purpose |
|---|---|
| `data_collector.py` | Reads power samples from the ADC via SPI on the Raspberry Pi |
| `feature_extraction.py` | Extracts the 43-feature vector (statistical + FFT + packet-rate proxies) from raw power windows |
| `prepare_cnn_data.py` | Builds the windowed, normalized dataset for the 1D-CNN |
| `train_model.py` | Trains the classical ML baselines |
| `multi_model_comparison.py` | Trains and compares XGBoost, Random Forest, SVM, and CNN |
| `cnn_model.py` | 1D-CNN architecture, training loop (fixed seed for reproducibility) |
| `dragon_slice_full_ablation.py`, `dragon_slice_best_config_gdrfar.py`, `dragon_slice_replication_ethernet_fixed.py` | Unsupervised convolutional autoencoder ("Dragon_Slice") replication and GDR-vs-FAR analysis |
| `shap_analysis.py` | SHAP feature-importance analysis (XGBoost) |
| `noise_robustness_test.py` | AWGN robustness sweep on raw signal windows |
| `chronological_split_test.py` | Time-ordered train/test split (leakage check) |
| `multi_seed_stability.py` | Multi-seed statistical significance testing |
| `ablation_ethernet_fixed.py`, `roc_and_ci_ethernet_fixed.py`, `final_analysis_ethernet_fixed.py`, `cae_per_class_breakdown.py`, `attack_transition.py` | Supporting robustness/validation analyses |
| `rf_onnx_export.py`, `xgb_onnx_export_ethernet_fixed.py`, `svm_onnx_export_ethernet_fixed.py`, `convert_to_onnx.py`, `verify_onnx.py` | Export trained models to ONNX for on-device inference |
| `rpi_inference.py` | Runs the exported ONNX model live on the Raspberry Pi |
| `generate_paper_figures.py`, `analysis_plots.py`, `replot_gdrfar_english.py` | Generate the manuscript's figures |
| `explore.py`, `final_analysis.py` | Exploratory analysis / early-pipeline results |

## Installation

```bash
pip install -r requirements.txt
```

`spidev` is only required on the Raspberry Pi itself (SPI communication
with the MCP3008); it is not needed to run the offline training/analysis
scripts on a regular PC.

## Data

The raw power-trace dataset, extracted feature matrices, and trained
model weights are archived separately on Zenodo (not included in this
repository due to size):

**[TODO: separate Zenodo dataset DOI, not yet uploaded]**

## Citation

This code itself is archived and citable via Zenodo:

[https://doi.org/10.5281/zenodo.22864415](https://doi.org/10.5281/zenodo.22864415)

If you use this code, please also cite the associated article once
published (see `CITATION.cff`, or use GitHub's "Cite this repository"
button):

**[TODO: paper citation once accepted/published]**

## License

MIT License — see [`LICENSE`](LICENSE). Free to use, modify, and
redistribute; please cite the associated article (above) if you do.
