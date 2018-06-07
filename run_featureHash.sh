# 定义图片格式
ORIGN_IMAGE_FORMAT=".webp"
AUG_IMAGE_FORMAT=".jpg"

# 定义原始文件路径
ORIGN_IMAGE_PATH="/home/lol/DeepLearn/混合"
AUG_IMAGE_PATH=${ORIGN_IMAGE_PATH}_加强数据集
TIDY_AUG_IMAGE_PATH=${AUG_IMAGE_PATH}_整理版

echo -e "\033[47;31m                                                    \033[0m" 
echo -e "\033[47;31m               开始进行[数据加强]                   \033[0m" 
echo -e "\033[47;31m                                                    \033[0m" 

python ./data_augmentation.py -f $ORIGN_IMAGE_PATH -s $AUG_IMAGE_PATH -t $ORIGN_IMAGE_FORMAT

echo -e "\033[47;31m                                                    \033[0m" 
echo -e "\033[47;31m               开始进行[规整数据]                   \033[0m" 
echo -e "\033[47;31m                                                    \033[0m" 

python ./tidy_image.py -f $AUG_IMAGE_PATH -s $TIDY_AUG_IMAGE_PATH -t $AUG_IMAGE_FORMAT

echo -e "\033[47;31m                                                    \033[0m" 
echo -e "\033[47;31m               开始进行[提取特征]                   \033[0m" 
echo -e "\033[47;31m                                                    \033[0m" 

python ./use_mxnet_get_feature.py -f $TIDY_AUG_IMAGE_PATH

echo -e "\033[47;31m                                                    \033[0m" 
echo -e "\033[47;31m               开始进行[提取测试]                   \033[0m" 
echo -e "\033[47;31m                                                    \033[0m" 

python ./eval_featureHash.py -f $ORIGN_IMAGE_PATH -m $TIDY_AUG_IMAGE_PATH