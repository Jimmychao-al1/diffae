#python QATcode/sample_lora_intmodel.py --num_steps 100 --eval_samples 5000 --mode int
#python QATcode/sample_lora_intmodel.py --num_steps 20 --eval_samples 5000 --mode int
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Res --cache_threshold 0.03 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Res --cache_threshold 0.05 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Res --cache_threshold 0.075 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Res --cache_threshold 0.1 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Att --cache_threshold 0.03 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Att --cache_threshold 0.05 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Res --cache_threshold 0.01 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Att --cache_threshold 0.075 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Att --cache_threshold 0.1 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Res --cache_threshold 0.01 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Res --cache_threshold 0.03 --eval_samples 5000 --num_steps 20
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Res --cache_threshold 0.04 --eval_samples 5000 --num_steps 20
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Att --cache_threshold 0.04 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Att --cache_threshold 0.045 --eval_samples 5000 --num_steps 100
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Att --cache_threshold 0.075 --eval_samples 5000 --num_steps 20
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Att --cache_threshold 0.05 --eval_samples 5000 --num_steps 20
#
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Res --cache_threshold 0.02 --eval_samples 5000 --num_steps 20
#python QATcode/sample_lora_intmodel.py --mode int --enable_cache --cache_method Res --cache_threshold 0.03 --eval_samples 5000 --num_steps 100
python QATcode/sample_lora_intmodel.py --mode float --eval_samples 5000 --num_steps 100 --log_file QATcode/sample_100_5k.log
python QATcode/sample_lora_intmodel.py --mode float --eval_samples 5000 --num_steps 20 --log_file QATcode/sample_20_5k.log
python QATcode/sample_lora_intmodel.py --mode float --eval_samples 50000 --num_steps 100 --log_file QATcode/sample_100_50k.log
python QATcode/sample_lora_intmodel.py --mode float --eval_samples 50000 --num_steps 20 --log_file QATcode/sample_20_50k.log
python QATcode/sample_lora_intmodel.py --mode int --eval_samples 100 --num_steps 100 --log_file QATcode/sample_100_5k_int.log
python QATcode/sample_lora_intmodel.py --mode int --eval_samples 100 --num_steps 20 --log_file QATcode/sample_20_5k_int.log