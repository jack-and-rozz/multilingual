#!/bin/bash

# List of arguments that can be dynamically specified at runtime.
# e.g. ./run.sh checkpoint/tmp test --test_data_path=dataset/test.another
params_list=(log_file interactive cleanup train_data_path test_data_path evaluate_data_path debug batch_size vocab_size train_data_size)

usage() {
    echo "Usage:$0 checkpoint_path mode [config_path]"
    exit 1
}

# Parse dynamic arguments. 
source ./scripts/manually_getopt.sh $@

if [ $argc -lt 2 ];then
    usage;
fi
bin_file=bins/wikiP2D.py
checkpoint_path=${argv[0]}
mode=${argv[1]}
config_path=${argv[2]}
config_file=config
log_file=$checkpoint_path/${mode}.log

# Use the existing config in the checkpoint directory if there is, otherwise use configs/config as default.
if [ "${config_path}" = "" ]; then
    if [ -e ${checkpoint_path}/${config_file} ]; then
	config_path=${checkpoint_path}/${config_file}
    else
	config_path=configs/config
	#echo "specify config file when start training from scratch."
	#exit 1
    fi
fi

# Prioritize runtime options (parsed from 'scripts/manually_optget.sh')
if [ ${#opt_names[@]} != '0' ]; then
    for i in $(seq 0 $(expr ${#opt_names[@]} - 1)); do
	name=${opt_names[$i]}
	value=${opt_values[$i]}
	eval $name=$value
    done;
fi

params=""
for param in ${params_list[@]}; do 
    if [ ! ${!param} = "" ]; then
	params="${params} --${param}=${!param}"
    fi
done;

python src/main.py $checkpoint_path $mode $config_path $params
