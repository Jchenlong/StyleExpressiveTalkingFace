set -e

efs_model_names1=(
AZ3bkD46
5Iob5FSx
KLhKUOMV
3gkFXTXu
WWssO9tm
b1e3Qmux_3
5vtehuo2
0uLSE4ZB
26u6M63q
TV1UKD0r
)



efs_model_names=(
${efs_model_names1[@]}
)

for efs_model_name in ${efs_model_names[@]}
do
  echo ${efs_model_name}
  video_path=/data1/chenlong/online_model_set/video/${efs_model_name}/face.mp4
  dst_dir=/data1/chenlong/online_model_set/face/${efs_model_name}/
  mkdir -p ${dst_dir}
  CUDA_VISIBLE_DEVICES=0 python tools/process_video_clip.py \
    --video_path ${video_path} \
    --dst_dir ${dst_dir} \

done