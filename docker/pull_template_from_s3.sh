set -e

#efs_model_names1=(
#lady9_s0_20240118
##man3_s0_20240118
#lady9_s4_20240118
##ruying7_s0_0117
#lady8_20240118
##heygen2_s0_0124
#man3_s1_20240118
#lady9_s1_20240118
##neuralavatar6_s0_0117
##neuralavatar5_s0_0117
##ruying2_s0_0116
##heygen3_s0_0124
##ruying3_s0_0116
##wondershare4_s0_0122
##Kevin_s0_0124
##neuralavatar1_s0_0117
##ruying4_s0_0116
##heygen6_s0_0124
##Alice_s0_0117
#man3_s2_20240118
##neuralavatar3_s0_0117
#lady9_s2_20240118
##neuralavatar2_s0_0122
##ruying9_s0_0117
#lady9_s3_20240118
##ruying10_s0_0117
##wondershare7_s0_0122
##wondershare2_s0_0122
##ruying11_s0_0117
##heygen1_s1_0124
##ruying1_s0_0116
##ruying5_s0_0116
##Adam_s0_0124
##neuralavatar7_s0_0117
##heygen4_s0_0124
##neuralavatar3_s1_0117
#man3_s3_20240118
##wondershare6_s0_0122
##heygen1_s0_0122
#)

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
  aws s3 cp s3://update-weights/model/Voice2Lip_v1.3.1/${efs_model_name}/templates/face.mp4  /data1/chenlong/online_model_set/video/${efs_model_name}/
done