MODELS_PATH="/home/slu/tasks/models_02/research/slim"
TRAIN_DIR="/www/"


cd $MODELS_PATH

python train_image_classifier.py \
    --train_dir=/www/clothes_train_tf/logs \
    --dataset_dir=/www/clothes_train_tf \
    --num_samples=481448 \
    --num_classes=100000 \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=/home/slu/tasks/inception_resnet_v2_2016_08_30.ckpt \
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
    --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
    --learning_rate=0.01