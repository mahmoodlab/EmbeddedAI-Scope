### TOY Example
First download [data](https://drive.google.com/drive/folders/1ofAFtL0PJd7fewOs3b44ejUzM-FwyQk0?usp=sharing) and put the data folder bwh_lung_2652 inside **phone_clam/data/**

See [here](INSTALLATION.md) for installation.

To run example:
``` shell
CUDA_VISIBLE_DEVICES=0 python infer_and_interpret.py --config_file heatmap_config_test.yaml --overlap 0.5
```

Generated heatmaps will be saved under **heatmaps/heatmap_production_results**.
Inferred results will be saved under **heatmaps/results**.
Settings can be modified using .yaml config files stored under **heatmaps/configs**; the config file corresponding the toy example is **heatmap_config_test.yaml** and is passed to the inference script via --config_file