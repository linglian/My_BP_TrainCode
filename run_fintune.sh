MODELS_PATH="/home/slu/tasks/models_02/research/slim"
TRAIN_DIR="/www/"


cd $MODELS_PATH

python train_image_classifier.py \
    --train_dir=/www/clothes_train_tf/logs \
    --dataset_dir=/www/clothes_train_tf \
    --num_samples=481448 \
    --num_classes=100000 \
    --model_name=resnet_v1_50 \
    --checkpoint_path=/home/slu/tasks/resnet_v1_50.ckpt \
    --checkpoint_exclude_scopes=resnet_v1_50/logits \
    --learning_rate=0.01