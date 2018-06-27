MODELS_PATH="/home/slu/tasks/models_02/research/slim"
TRAIN_DIR="/www/linglian_test_data"

cd $MODELS_PATH

python train_image_classifier.py \
    --train_dir=${TRAIN_DIR}/logs \
    --dataset_dir=${TRAIN_DIR} \
    --dataset_split_name=list_tfrecord \
    --num_samples=43769 \
    --num_classes=100000 \
    --batch_size=32 \
    --model_name=resnet_v1_50 \
    --checkpoint_exclude_scopes=resnet_v1_50/logits \
    --learning_rate=0.01