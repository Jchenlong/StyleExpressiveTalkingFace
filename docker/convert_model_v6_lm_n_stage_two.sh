expname=$1
decoder_path=$2
image_path=$3
to_path=$4
patch=0


python ./tools/deploy_lm_n_two_stage.py \
--exp_name ${expname} \
--decoder_path ${decoder_path} \
--to_path ${to_path} \
--image_path  ${image_path} \
--patch ${patch}