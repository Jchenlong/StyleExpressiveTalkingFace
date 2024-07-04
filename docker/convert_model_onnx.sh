expname=$1
decoder_path=/data1/chenlong/online_model_set/exp_ori/${expname}/results/pti_ft_512/snapshots/100.pth
echo ${expname}
echo ${decoder_path}
docker run  \
       -v `pwd`:/app/ \
       -v /data1:/data1 \
       -v /data1/wanghaoran/Amemori:/data1/wanghaoran/Amemori \
       -w /app \
       --rm \
       -it deep_engine:pytorch-113 \
       python ./tools/deploy.py \
       --exp_name $expname\
       --decoder_path \
       $decoder_path \
      --to_path /app/release_public \
      --pwd /data1/wanghaoran/Amemori/ExpressiveVideoStyleGanEncoding