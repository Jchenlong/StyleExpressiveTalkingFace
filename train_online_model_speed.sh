#!/bin/bash
set -e
CUDA_DEVICES=5

efs_model_names1=(
man3_s3_20240118
man3_s2_20240118
man3_s1_20240118
lady9_s4_20240118
lady9_s1_20240118
lady9_s2_20240118
)


efs_model_names=(
${efs_model_names1[@]}
)

train_pose_facial_dir=/data1/chenlong/local_code/expressive_talkinghead_encoding-main
ft_pose_dir=/data1/chenlong/github/expressive_talkinghead_encoding
train_offset_dir=/data1/chenlong/github/StyleExpressiveTalkingFace

for efs_model_name in ${efs_model_names[@]}
do
  train_data=/data1/chenlong/online_model_set/face/${efs_model_name}/face
  exp_dir=/data1/chenlong/online_model_set/exp_speed_v2/${efs_model_name}/results
  cd ${train_pose_facial_dir}
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} source ${train_pose_facial_dir}/sh_scripts/train_speed_pose_v2/train.sh ${efs_model_name}
#  cd ${ft_pose_dir}
#  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} source ${ft_pose_dir}/scripts/1_public_pose_finetuning_027/train.sh ${train_data} ${exp_dir}
#  cd ${train_offset_dir}
#  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} source ${train_offset_dir}/online_model_set_get_data.sh ${efs_model_name}
#  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} source ${train_offset_dir}/scripts/speed_data/scripts/${efs_model_name}/train.sh ${CUDA_DEVICES} ${efs_model_name}
#  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} source ${train_offset_dir}/docker/convert_model.sh ${efs_model_name}
done