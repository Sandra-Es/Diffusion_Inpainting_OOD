

PL_MODEL_PATH="weights/best_512.ckpt"
CONFIG_PATH="/mnt/MAE-FAR/configs/config.yml"       #Config File path
IMAGE_PATH="/mnt/data/celeba/images/gt"             #GT Image path
MASK_PATH="/mnt/data/celeba/masks/semantic/nose"    #Mask path
OUTPUT_PATH="./results/semantic/nose"               #Save directory
TEST_IMG_SCALE=512


CUDA_VISIBLE_DEVICES=0 python test_custom.py \
  --resume ${PL_MODEL_PATH} \
  --config ${CONFIG_PATH} \
  --image_path ${IMAGE_PATH} \
  --mask_path ${MASK_PATH} \
  --output_path ${OUTPUT_PATH} \
  --image_size ${TEST_IMG_SCALE} \
  --load_pl