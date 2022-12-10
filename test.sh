# Experiment all tricks without center loss without re-ranking: 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on (=raw all trick, softmax_triplet.yml)
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss
# without re-ranking
python3 tools/test.py \
--config_file='configs/softmax_triplet.yml' \
MODEL.DEVICE_ID "('1')" \
DATASETS.NAMES "('market1501')"  \
DATASETS.ROOT_DIR "('/mnt/CEPH_YIGO4/perception/g0/weidong.shi/workspace/ReID')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('/mnt/CEPH_YIGO4/perception/g0/weidong.shi/project/Baseline/results/Market/baseline/resnet50_model_42.pth')" \
OUTPUT_DIR "('/mnt/CEPH_YIGO4/perception/g0/weidong.shi/project/Baseline/results/Market/baseline/test')"