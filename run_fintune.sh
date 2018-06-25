MODELS_PATH="/home/slu/tasks/models_02/research/slim"
TRAIN_DIR="/www/clothes_train_tf"


cd $MODELS_PATH

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR}/logs \
    --dataset_dir=${TRAIN_DIR} \
    --num_samples=43769 \
    --num_classes=100000 \
    --model_name=resnet_v1_50 \
    --checkpoint_path=/home/slu/tasks/resnet_v1_50.ckpt \
    --checkpoint_exclude_scopes=resnet_v1_50/logits \
    --learning_rate=0.01