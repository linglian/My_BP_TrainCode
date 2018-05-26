MODELS_PATH="/home/slu/tasks/models_02/research/slim"
TRAIN_DIR="/www/clothes_train_tf"

cd $MODELS_PATH

python eval_image_classifier.py \
    --dataset_dir=${TRAIN_DIR} \
    --dataset_name=buliao \
    --num_samples=481448 \
    --num_classes=100000 \
    --model_name=resnet_v1_50 \
    --checkpoint_path=${TRAIN_DIR}/logs \