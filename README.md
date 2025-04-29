# DRCR2Net
Enhancing Defocus Blur Detection through Dual Recurrent Complementary Residual Refinemen by Longrui Li,Liqing Huang,Tianqiang Huang
and Haifeng Luo is submitted to The Visual Computer

## 1.Requirement

Python --- 3.9

Pytorch --- 2.1.0

Nvidia Geforce RTX 3060(12G)

## 2. Dataset
- `train_data`: The data for training.
  - `1204source`: Contains 604 training images of CUHK Dataset and 600 training images of DUT Dataset, 1204 in total.
- `test_data`: The data for testing.
  - `CUHK`: Contains 100 testing images of CUHK Dataset and it's GT.
  - `DUT`: Contains 400 testing images of DUT Dataset and it's GT.
   
## 3. Train
You can use the following command to test：
> python train.py
- `train.py`: the entry point for training.
- `backbone/`: contains pretrained backbone network.
- `network/`: contains models and main modules.
- `dataset.py`: process the dataset before training.

## 4. Test
You can use the following command to test：
> python test.py

## 5. Eval
You can use the following command to evaluate the results：
> python metrics.py

## 6.Citation
> @article{Li2025,
>  author       = {Longrui Li and
>                  Liqing Huang and
>                  Tianqiang HUang and
>                 Haifeng Luo},
>  title        = {Enhancing Defocus Blur Detection through Dual Recurrent Complementary Residual Refinement},
>  journal      = {The Visual Computer}
> }

