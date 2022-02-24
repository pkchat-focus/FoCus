#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
#BART base MTL
#CUDA_VISIBLE_DEVICES=0 nohup python inference_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10/ --filename bart_base_20_new > infer_log_focus/BART_base_inference_20.log &
#CUDA_VISIBLE_DEVICES=1 nohup python inference_test.py --model_name BART --max_length 30 --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10/ --filename bart_base_30 > infer_log_focus/BART_base_inference_30.log &

#GPT2 base MTL
#CUDA_VISIBLE_DEVICES=2 nohup python inference_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10/ --filename gpt2_small_20 > infer_log_focus/GPT2_small_inference_20.log &
#CUDA_VISIBLE_DEVICES=3 nohup python inference_test.py --model_name GPT2 --max_length 30 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10/ --filename gpt2_small_30_new > infer_log_focus/GPT2_small_inference_30_new.log &

#BART base LM
#CUDA_VISIBLE_DEVICES=0 nohup python inference_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_LM/ --filename bart_base_20_LM_new > infer_log_focus/BART_base_inference_20_LM.log &
#CUDA_VISIBLE_DEVICES=1 nohup python inference_test.py --model_name BART --max_length 30 --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_LM/ --filename bart_base_30_LM > infer_log_focus/BART_base_inference_30_LM.log &

#GPT2 base LM
#CUDA_VISIBLE_DEVICES=2 nohup python inference_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_LM/ --filename gpt2_small_20_LM > infer_log_focus/GPT2_small_inference_20_LM.log &
#CUDA_VISIBLE_DEVICES=4 nohup python inference_test.py --model_name GPT2 --max_length 30 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_LM/ --filename gpt2_small_30_LM_new > infer_log_focus/GPT2_small_inference_30_LM_new.log &

#BART base woPS
#CUDA_VISIBLE_DEVICES=0 nohup python inference_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_PS/ --filename bart_base_20_woPS_new > infer_log_focus/BART_base_inference_20_woPS.log &
#CUDA_VISIBLE_DEVICES=1 nohup python inference_test.py --model_name BART --max_length 30 --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_PS/ --filename bart_base_30_woPS > infer_log_focus/BART_base_inference_30_woPS.log &

#GPT2 base woPS
#CUDA_VISIBLE_DEVICES=2 nohup python inference_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_PS/ --filename gpt2_small_20_woPS > infer_log_focus/GPT2_small_inference_20_woPS.log &
#CUDA_VISIBLE_DEVICES=5 nohup python inference_test.py --model_name GPT2 --max_length 30 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_PS/ --filename gpt2_small_30_woPS_new > infer_log_focus/GPT2_small_inference_30_woPS_new.log &

#BART base woKS
#CUDA_VISIBLE_DEVICES=0 nohup python inference_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_KS/ --filename bart_base_20_woKS > infer_log_focus/BART_base_inference_20_woKS.log &
#CUDA_VISIBLE_DEVICES=1 nohup python inference_test.py --model_name BART --max_length 30 --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_KS/ --filename bart_base_30_woKS > infer_log_focus/BART_base_inference_30_woKS.log &

#GPT2 base woKS
#CUDA_VISIBLE_DEVICES=2 nohup python inference_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_KS/ --filename gpt2_small_20_woKS > infer_log_focus/GPT2_small_inference_20_woKS.log &
#CUDA_VISIBLE_DEVICES=6 nohup python inference_test.py --model_name GPT2 --max_length 30 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_KS/ --filename gpt2_small_30_woKS_new > infer_log_focus/GPT2_small_inference_30_woKS_new.log &



echo
