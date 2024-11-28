set -e
save_path=$2
config_path=$1
python_file_path=$3
function main
{
    python ${python_file_path}/TalkingFace/infer_v7.py \
                           --config_path ${config_path} \
                           --save_path  ${save_path}
}
main
