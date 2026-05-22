# ERIC-edge

Rain sensing on the edge using traditional ML workflows (visual + audio features).

## Configuration

All I/O paths and controls are defined in `control.yaml`.

## ROI preparation (visual features)

Find region of interest for calculating visual features:

```bash
python 1_cal_avg_intensity.py -d /path/to/videos/findROI/day/ -c Avg_gray_df_0815_13-16pm
```

## Processing pipeline

1) Extract visual features:

```console
python 1_process_visual_features.py -d video_dir/
```

If you already have bounding boxes, fetch ROIs from `boxes.json`:

```console
python 1_process_visual_features_boxes.py -d /path/to/videos/train/ -b /path/to/boxes/boxes.json
```

2) Extract audio features:

Extract wav files from mp4 files first:

```console
python 2_process_audio_features.py -d video_dir/ -w
```

Use existing wav files:

```console
python 2_process_audio_features.py -d video_dir/
```

3) Aggregate visual and audio features with labels (day/night, `is_rain`, `rain_amount`).
Update input and output paths in `control.yaml` as needed.

```console
python 3_1_aggregate_features.py
```

4) Run rain detection with classification models.
Use `split_point` and `end_point` to control train/test windows.

```console
python 4_rain_class.py
```

5) Run rain estimation (regression):

```console
python 5_rain_regress.py
```

6) Plot rain detection results for selected days:

```console
python 6_plot_israin.py
```

7) Plot rainfall estimation results.
First split estimated rainfall data by date into separate csv files, then plot:

```console
python 7_plot_rainfall.py
```

8) Calculate regression scores based on filtered rainfall values from detection results:

```console
python 8_cal_regress_score.py
```