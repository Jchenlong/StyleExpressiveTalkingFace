#!/bin/bash
set -e
ROOT_PATH=$1

exp_name='exp'
#file_dirname=$(dirname ${ROOT_PATH})
file_dirname=${ROOT_PATH}
folder_name=$(basename "$file_dirname")
python_file_path=$3

cd ${python_file_path}

echo ${folder_name}
#mkdir xxx
directory=${file_dirname}/lm_train_n/datasets/${folder_name}
directory_ori=${file_dirname}/lm_train/datasets/${folder_name}
mkdir -p ${directory}

gt_smooth=$2
# get id.pt
cp -f ${ROOT_PATH}/${exp_name}/cache.pt ${directory}/id.pt

# get face landmark
python ${python_file_path}/get_landmarks_n.py --from_path ${gt_smooth}  \
                        --to_path ${directory}/lm3d.npy

#lm3d_dir_path=$(dirname ${gt_smooth})
#cp -f ${lm3d_dir_path}/lm2d.npy ${directory}/lm3d.npy
#echo ${lm3d_dir_path}/lm2d.npy
#cp -f ${directory_ori}/lm3d.npy ${directory}/lm3d.npy

# get id landmark
python ${python_file_path}/get_id_landmarks.py --id_path ${directory}/id.pt \
                           --landmark_path ${directory}/lm3d.npy \
                           --to_path ${directory}/id_landmark.npy
# get pose pt
python ${python_file_path}/merge_more2one.py ${ROOT_PATH}/${exp_name}/pose ${directory}/pose.pt

# get attribute pt
python ${python_file_path}/merge_more2one.py ${ROOT_PATH}/${exp_name}/expressive ${directory}/attribute.pt

# get train/val data
# attribute
python ${python_file_path}/tools/get_validate_data.py --from_path ${directory}/attribute.pt \
                                  --to_path ${directory}/ \
                                  --ratio 0.9

# landmark
python ${python_file_path}/tools/get_validate_data.py --from_path ${directory}/lm3d.npy \
                                  --to_path ${directory} \
                                  --ratio 0.9
## get training scripts
mkdir -p ${file_dirname}/lm_train_n/scripts/${folder_name}
cp -r ${python_file_path}/scripts/speed_set_template_v4/train.sh ${file_dirname}/lm_train_n/scripts/${folder_name}/

pose_path=${directory}/pose.pt
attr_path=${directory}/attribute.pt
id_path=${directory}/id.pt
id_landmark_path=${directory}/id_landmark.npy
f_space_path=${ROOT_PATH}/${exp_name}/f_space.pt

attr_train_path=${directory}/train_attr.pt
attr_val_path=${directory}/val_attr.pt

ldm_train_path=${directory}/train_landmarks.npy
ldm_val_path=${directory}/val_landmarks.npy

sed --expression "s@%DATASET%@${directory}@" \
    -e "s@%F_SPACE_PATH%@${f_space_path}@g" \
    ${python_file_path}/scripts/speed_set_template_v4/config.yaml \
    > ${file_dirname}/lm_train_n/scripts/${folder_name}/config.yaml

# get config_test.yaml

pose_latent_path=${ROOT_PATH}/${exp_name}/pose
train_config_path=${file_dirname}/lm_train_n/scripts/${folder_name}/config.yaml
train_weight_path=${file_dirname}/lm_train_n/results/${folder_name}/snapshots/best.pth
pti_path=${ROOT_PATH}/${exp_name}/pti_ft_512/snapshots
f_space_decoder_path=${ROOT_PATH}/${exp_name}/pti_ft_512/snapshots
video_landmark_path=${directory}/lm3d.npy
driving_images_dir=${gt_smooth}

sed --expression "s@%POSE_LATENT_PATH%@${pose_latent_path}@" \
    -e "s@%TRAIN_CONFIG_PATH%@${train_config_path}@g" \
    -e "s@%TRAIN_WEIGHT_PATH%@${train_weight_path}@g" \
    -e "s@%PTI_WEIGHT_PATH%@${pti_path}@g" \
    -e "s@%ATTRIBUTE_PATH%@${attr_path}@g" \
    -e "s@%ID_PATH%@${id_path}@g" \
    -e "s@%LANDMARK_PATH%@${video_landmark_path}@g" \
    -e "s@%ID_LANDMARK_PATH%@${id_landmark_path}@g" \
    -e "s@%GT_PATH%@${driving_images_dir}@g" \
    -e "s@%F_SPACE_PATH%@${f_space_path}@g" \
    ${python_file_path}/scripts/speed_set_template_v4/config_test_n.yaml \
    > ${file_dirname}/lm_train_n/scripts/${folder_name}/config_test.yaml

config_path=${file_dirname}/lm_train_n/scripts/${folder_name}/config.yaml
config_test_path=${file_dirname}/lm_train_n/scripts/${folder_name}/config_test.yaml
results_path=${file_dirname}/lm_train_n/results/${folder_name}
mkdir -p ${results_path}

source ${file_dirname}/lm_train_n/scripts/${folder_name}/train.sh ${config_path} ${results_path} ${python_file_path}
source ${python_file_path}/scripts/infer_speed_v4.sh ${config_test_path} ${results_path} ${python_file_path}
source ${python_file_path}/scripts/infer_speed_v7.sh ${config_test_path} ${results_path} ${python_file_path}


python ./tools/deploy_lm_n_two_stage.py \
--exp_name ${folder_name} \
--decoder_path ${f_space_decoder_path} \
--to_path ${file_dirname}/lm_train_n \
--image_path  ${gt_smooth} \
--patch 0


