# Experiment all tricks without center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss

python3 tools/train.py \
--config_file='configs/softmax_triplet.yml' \
MODEL.DEVICE_ID "('4')" \
MODEL.NAME "('resnet50')" \
MODEL.PRETRAIN_PATH "('/home/weidong.shi1/data/resnet50-19c8e357.pth')" \
DATASETS.NAMES "('dukemtmc')" \
DATASETS.ROOT_DIR "('/home/weidong.shi1/data')" \
OUTPUT_DIR "('./results/baseline-res-50/Duke')"