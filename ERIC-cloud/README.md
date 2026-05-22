# ERIC-cloud

ResNet-based rain sensing pipeline.

## Data preprocessing

Videos are sliced into images every 5 seconds using OpenCV.

Slice RGB or DIF images (or both) with the `-f` option:

```console
python ImageSlicer.py -i video_dir -o image_dir -s video_start -e video_end -f rgb
```

Example:

```console
python ImageSlicer.py -i /path/to/videos/train/ -o /path/to/images/train_rgb/ -f rgb
```

Separate images into rain and norain folders based on labels:

```console
python ImageSeparator.py -d img_dir -f output_dir
```

Example:

```console
python ImageSeparator.py -d /path/to/images/train_rgb/ -f /path/to/images/train_rgb/
```

## Rain detection (classification)

Train ResNet18 from scratch:

```console
python train.py
```

Evaluate the best model on val and test sets:

```console
python eval.py
```

Test the trained model on sample images:

```console
python test.py
```

## Rainfall estimation (regression)

Generate rainfall amount labels for regression:

```console
python generate_rainfall_labels.py -d /path/to/images/train_rgb/ -f ../ -o rainfall_label_train.csv
```

Generate labels at 5s resolution (results are worse than 1-min resolution due to noise from finer labels):

```console
python generate_rainfall_labels_finer-inten.py -d /path/to/images/train_rgb/ -f ../ -o rainfall_label_train_5s.csv
```

Train the regressor:

```console
python train_regression.py
```

Evaluate and test the regressor:

```console
python eval_regression.py
```

Plot rainfall and compute minute averages (splits data by date):

```console
python plot_rainfall.py
```

Calculate filtered regression scores:

```console
python cal_score_filtered.py
```

## Visualization

Plot accuracy and loss:

```console
python plot_accuracy_loss.py
```

Plot rain detection results:

```console
python plot_israin.py
```
