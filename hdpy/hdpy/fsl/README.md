# Folk from: Few-shot Image Classification: Just Use a Libraryof Pre-trained Feature Extractors and a Simple Classifier: [**paper**](https://arxiv.org/pdf/2101.00562.pdf)

## Use the following links to download the data:

1. ILSVRC2012:
Register at [**ImageNet**](http://www.image-net.org/) and request for a username and an access key to download ILSRVC-2012 data set.

2. CUB-200-2011 Birds:
[**Birds**](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

3. FGVC-Aircraft:
[**Aircraft**](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)

4. FC100:
[**FC100**](https://drive.google.com/drive/folders/1nz_ADBblmrg-qs-8zFU3v6C5WSwQnQm6)

5. Omniglot:
[**Omniglot**](https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip)

6. Texture:
[**Texture**](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)

7. Traffic Sign:
[**Traffic Sign**](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip)

8. FGCVx Fungi:
[**Fungi**](https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz)
[**Annotations**](https://labs.gbif.org/fgvcx/2018/train_val_annotations.tgz)

9. Quick Draw:
[**Quick Draw**](https://console.cloud.google.com/storage/quickdraw_dataset/full/numpy_bitmap)
- Use [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install) to download the data:
```
gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy data/quickdraw
```
- Convert numpy files to jpeg images using  this [conversion code](https://github.com/C-Aniruddh/RapidDraw/blob/in-dev/processing/process_all.py)

10. VGG Flower:
[**VGG Flower**](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz),
[**Labels**](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat)


## Extracting Pretrained Library Features (PyTorch):

```
sh run_extract_pretrained_features.sh
```

## HD-based Few-shot Learning:

- Single library classifier example:
```
sh run_classifier_single.sh
```
