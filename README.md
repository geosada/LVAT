# LVAT
The code used in the evaluation of Latent Space Virtual Adversarial Training.

## Reference
This code is written based on the original VAT implementation (see [here](https://github.com/takerum/vat_tf)) and the implementation for Glow is heavily based on [this code](https://github.com/kmkolasinski/deep-learning-notes/tree/3c7779ea0063896bb3a759efa3e52d173aaae94b/seminars/2018-10-Normalizing-Flows-NICE-RealNVP-GLOW). 

## Requirements
tensorflow-gpu 1.14

## Preparation 1. Create symbolic links

```
cd lvat/
# for LVAT-VAE
ln -s ../vae/VAE.py .
ln -s ../vae/config.py .
ln -s ../vae/util .
ln -s ../vae/out_VAE_SVHN/ out_VAE_SVHN
ln -s ../vae/out_VAE_SVHN_aug/ out_VAE_SVHN_aug
ln -s ../vae/out_VAE_CIFAR10/ out_VAE_CIFAR10
ln -s ../vae/out_VAE_CIFAR10_aug/ out_VAE_CIFAR10_aug

# for LVAT-Glow
ln -s ../glow/out/SVHN/w_128__step_22__scale_3__b_128/ out_Glow_SVHN
ln -s ../glow/out/SVHN_aug/w_128__step_22__scale_3__b_128/ out_Glow_SVHN_aug
ln -s ../glow/out/CIFAR10/w_128__step_22__scale_3__b_128/ out_Glow_CIFAR10
ln -s ../glow/out/CIFAR10_aug/w_128__step_22__scale_3__b_128/ out_Glow_CIFAR10_aug
```

## Preparation 2. Create tfrecords dataset

```
cd vae/util/
```
and
``` 
python svhn.py --data_dir=<YOUR_PATH>
```
or
```
python cifar10.py --data_dir=<YOUR_PATH>
```

## Preparation 3. Building transfomer(VAE or Glow)

For VAE,
```
cd vae
python build_AE.py
```
and for Glow,
```
cd glow
python main.py
```
For both, datasets are identified in config.py (for VAE) or config_glow.py (for Glow).



## Training Classifier with LVAT
```
cd lvat
```
and for example
```
python train_semisup.py  --data_set=svhn --num_epochs=200
--epoch_decay_start=80 --epsilon=1.5 --top_bn --method=lvat --log__dir=${dir}
--data__dir=/data/img/SVHN__labeled_1000  --num_iter_per_epoch=400
--batch_size=32 --ul_batch_size=128  --num_labeled_examples=1000
--is_aug=True --ae_type=Glow
```
For SVHN, `--top_bn` option is necessary to achieve reported accuracy.
