set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results

function main
{
    CUDA_VISIBLE_DEVICES=6 python -m TalkingFace \
                           --config_path /data1/chenlong/github/StyleExpressiveTalkingFace/scripts/lm_train_data/scripts/man3_s0_20240118/config.yaml \
                           --save_path /data1/chenlong/0517/public_model_lm_train_data/results/${exp_name}_25
}

if [ ! -d "log/${exp_name}" ]; then
    
    mkdir -p "log/${exp_name}"
fi

_timestamp=`date +%Y%m%d%H`

main 0
