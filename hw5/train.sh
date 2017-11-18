#!/bin/bash

cd /home/dlewis/mt-class/en600.468/hw5

source activate p27

export n_gpus=`lspci | grep -i "nvidia" | wc -l`
export device=`nvidia-smi | sed -e '1,/Processes/d' | tail -n+3 | head -n-1 | perl -ne 'next unless /^\|\s+(\d)\s+\d+/; $a{$1}++; for(my $i=0;$i<$ENV{"n_gpus"};$i++) { if (!defined($a{$i})) { print $i."\n"; last; }}' | tail -n 1`
echo $device
python train.py --data_file hw5 --src_lang words --trg_lang phoneme --model_file modeldump --optimizer Adam -lr 1e-2 --batch_size 48 --gpuid $device


