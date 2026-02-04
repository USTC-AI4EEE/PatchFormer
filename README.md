# A Novel Patch-Based Transformer for Accurate Remaining Useful Life Prediction of Lithium-Ion Batteries

> **Authors:**
Lei Liu, Jiahui Huang, Hongwei Zhao, Tianqi Li, Bin Li.

This repo contains the code and data from our paper published in Journal of Power Sources [A Novel Patch-Based Transformer for Accurate Remaining Useful Life Prediction of Lithium-Ion Batteries](Paper: https://www.sciencedirect.com/science/article/pii/S0378775325000230).

## 1. Abstract

Accurate prediction of the remaining useful life (RUL) of lithium-ion batteries is critical for ensuring their safe and reliable operation. Nevertheless, achieving precise RUL prediction presents significant challenges due to the intricate degradation mechanisms inherent in battery systems and the influence of operational noise, particularly the capacity regeneration phenomena. To address these issues, we propose a novel patch-based transformer (PatchFormer). The proposed architecture incorporates a Dual Patch-wise Attention Network (DPAN), which effectively captures global correlations between patches via inter-patch attention while also addressing local dependencies within individual patches through intra-patch attention. This dual attention mechanism facilitates the selection and integration of both global and local features. Additionally, we implement a Feature-wise Attention Network (FAN) that utilizes self-attention on the fused temporal features to elucidate the interrelationships among these features. This model proficiently integrates global associations and local details within the partitioned time series, enabling it to accurately delineate the capacity degradation trends while precisely capturing the capacity regeneration occurrences. Extensive experiments are conducted on three public battery degradation datasets, benchmarking our model against state-of-the-art time series forecasting (TSF) models. The results consistently show that our model achieves superior prediction performance across different prediction starting points.

## 2. Environment setup

- first method (recommended)

```bash
conda env create -f patchformer.yaml
conda activate patchformer
```

- second method

```bash
conda create -n patchformer python=3.10.13
conda activate patchformer
pip install torch==1.13.1
pip install -r requirements.txt
```

## 3. Datasets

The CALCE dataset is already placed in the datasets folder. The URLs of other datasets are as follows:

CALCE dataset: https://calce.umd.edu/battery-data.

NASA dataset: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository.

TJU dataset: https://github.com/wang-fujin/PINN4SOH/tree/main/data/TJU%20data/Dataset_3_NCM_NCA_battery.

The data preprocessing code is provided. Note: The main code related to RUL prediction is run on the CALCE dataset.  Modifications are required to run it on the NASA and TJU datasets.

## 4. Usage

- an example for train and evaluate a new model：

```bash
python RUL_Prediction_PatchFormer.py
```

- You can get the following output:

```bash
['encoder_cont']: torch.Size([128, 64, 2])
['decoder_cont']: torch.Size([128, 1, 2])
y: torch.Size([1])
model name:PatchFormer

selected battery name:CS2_35, start point:300

train dataset:2323,val:290,test:647
Input Feature num:2 ,name:['Capacity', 'target']
time_varying_unknown_reals num:1,decoder num:1 ,name:['target']
model name:PatchFormer

Number of parameters in network(参数数量(Params): 97.9k
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [2]

  | Name            | Type        | Params
------------------------------------------------
0 | loss            | SMAPE       | 0     
1 | logging_metrics | ModuleList  | 0     
2 | network         | PatchFormer | 97.9 K
------------------------------------------------
97.9 K    Trainable params
0         Non-trainable params
97.9 K    Total params
0.392     Total estimated model params size (MB)
Epoch 81: 100%|██████████| 21/21 [03:05<00:00,  8.82s/it, loss=0.00934, train_loss_step=0.00956, val_loss=0.0101, train_loss_epoch=0.00916]
Epoch 81: 100%|██████████| 21/21 [03:05<00:00,  8.82s/it, loss=0.00934, train_loss_step=0.00956, val_loss=0.0101, train_loss_epoch=0.00916]
best_model_path: results_CALCE_RUL_prediction_sl_64/CS2_35/PatchFormer/SP300/PatchFormerNetModel_2024-11-02 03-48-31/PatchFormerNetModel/checkpoints/epoch=81-step=1312.ckpt
best_model_path:results_CALCE_RUL_prediction_sl_64/CS2_35/PatchFormer/SP300/PatchFormerNetModel_2024-11-02 03-48-31/PatchFormerNetModel/checkpoints/epoch=81-step=1312.ckpt <_io.TextIOWrapper name='results_CALCE_RUL_prediction_sl_64/CS2_35/PatchFormer/SP300/PatchFormerNetModel_2024-11-02 03-48-31/log_Feas_2_2_in_l_64_out_l_1_Pcap.txt' mode='w' encoding='UTF-8'>
```

## 5. Citation

If you find our work useful in your research, please consider citing:

```latex
@article{liu2025patchformer,
  title={PatchFormer: A novel patch-based transformer for accurate remaining useful life prediction of lithium-ion batteries},
  author={Liu, Lei and Huang, Jiahui and Zhao, Hongwei and Li, Tianqi and Li, Bin},
  journal={Journal of Power Sources},
  volume={631},
  pages={236187},
  year={2025},
  publisher={Elsevier}
}
```

If you have any problems, contact me via liulei13@ustc.edu.cn.
