set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
username=`whoami`

mkdir -p results

exp_name=$2
echo $exp_name

function main
{
    CUDA_VISIBLE_DEVICES=$1 python -m TalkingFace \
                           --config_path /data1/chenlong/github/StyleExpressiveTalkingFace/scripts/speed_data/scripts/${exp_name}/config.yaml \
                           --save_path  /data1/chenlong/online_model_set/speed_data/results/${exp_name}
}

#if [ ! -d "log/${exp_name}" ]; then
#
#    mkdir -p "log/${exp_name}"
#fi

_timestamp=`date +%Y%m%d%H`

main 0
