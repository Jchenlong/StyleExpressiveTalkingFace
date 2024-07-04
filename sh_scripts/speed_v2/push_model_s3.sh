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

train_offset_dir=/data1/chenlong/github/StyleExpressiveTalkingFace

for efs_model_name in ${efs_model_names[@]}
do
  cd ${train_offset_dir}
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} source ${train_offset_dir}/docker/convert_model.sh ${efs_model_name}
done