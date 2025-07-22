
# PCGMMF

A Prognostic Method for Breast Cancer Recurrence and Metastasis Risk Prediction Based on Multimodal Feature Fusion.


## Environmental requirements
python(3.7.5)、torch(1.13.1+cu117)、torchvision(0.4.2)、pandas(1.3.5)、Pillow(6.2.1)、numpy(1.21.6)、opencv-python(4.1.1.26)、openslide-python(1.2.0)、openslides(3.3)、keras(2.11.0)




## Histopathological images recognition and segmentation

The first step focuses on the identification and segmentation of the tissue.The segmentation of specific slides can be adjusted by tuning the individual parameters. DATA_DIRECTORY is a directory of histopathological images. RESULTS_DIRECTORY is a directory of saving the segmentation results.
```python
  python create_patches.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 224 --stitch --seg --patch --patch_level
```
## Feature extracting by Vision-LSTM pre-trained model

The second step mainly involves using a Vision-LSTM pre-trained model to extract features from organ pathology images. CSV_FILE NAME is a segmentation parameter file. FEATURES_DIRECTORY is a directory of saving feature extraction results.

```python
 python Train_VIL.py --data_h5_dir RESULTS_DIRECTORY --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 256 --slide_ext .svs
```
## Dataset split

The third step is spliting the dataset. 

```python
 python create_splits.py --task Recurrence_metastasis_vs_normal --seed 1 --label_frac 1 --k 1
```
## Training

The fourth step is training. ./splits/Recurrence_metastasis_vs_normal_100_balance is a directory of saving the dataset splitting. Result is a directory of saving the training results.

```python
 ppython TN_main.py --early_stopping --lr 6e-6 --k 1 --split_dir ./splits/Recurrence_metastasis_vs_normal_100_balance --exp_code Result --task Recurrence_metastasis_vs_normal --data_root_dir FEATURES_DIRECTORY --drop_out
```
## Valuation

```python
 ppython Valuation.py
```