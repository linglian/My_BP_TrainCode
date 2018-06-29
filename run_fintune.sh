MODELS_PATH="/home/slu/tasks/models/research/slim"
TRAIN_DIR="/home1/alldata/tf_train"

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
    --learning_rate=0.0001 \
    --num_readers=16 \
    --num_preprocessing_threads=16 \
    --num_clones=4