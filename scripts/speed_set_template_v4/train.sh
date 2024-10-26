set -e
bash_script=`dirname ${0}`
exp_name=`echo $bash_script | awk -F '/' '{print $NF}'`
echo $exp_name
username=`whoami` 

mkdir -p results
config_path=$1
save_path=$2
python_file_path=$3
function main
{
#    python /opt/conda/lib/python3.9/site-packages/TalkingFaceTrainer/StyleExpressiveTalkingFace/TalkingFace/train_lm.py \
    python ${python_file_path}/TalkingFace/train_lm.py \
                           --config_path ${config_path} \
                           --save_path ${save_path}
}

_timestamp=`date +%Y%m%d%H`
main 0