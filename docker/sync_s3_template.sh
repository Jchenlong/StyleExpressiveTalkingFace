expname=$1
decoder_path=/data1/chenlong/online_model_set/exp_ori/${expname}/results/pti_ft_512/snapshots/100.pth
echo ${expname}
echo ${decoder_path}

cp /data1/chenlong/0517/public_model/config.yaml /data1/chenlong/online_model_set/release/$expname/config.yaml
aws s3 sync s3://update-weights/model/Voice2Lip_v1.3.1/soJHXhs8/templates  /data1/chenlong/online_model_set/release/$expname/templates
aws s3 sync /data1/chenlong/online_model_set/release/$expname  s3://update-weights/model/Voice2Lip_v1.3.1/${expname}_set_v1