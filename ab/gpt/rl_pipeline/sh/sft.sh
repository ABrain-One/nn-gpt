cd ../../../..
nohup python3 -m ab.gpt.TuneBackbone \
  --llm_conf backbone_sft_config.json \
  --sft_nn_prefixes rl-bb-struct1,rl-bb-struct1-v2 \
  --gen_nn_prefix rl-bb-struct1-v2-dscoder7b-sftcycle-full \
  --epoch_root out/sft_full/backbone_struct1_20260716 \
  --test_nn 9 \
  --num_cycles 5 \
  --sft_dataset cifar-10 > logs-full.txt 2>&1 &
tail -f logs-full.txt
