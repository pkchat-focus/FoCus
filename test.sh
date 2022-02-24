#(c) 2021 NCSOFT Corporation & Korea University. All rights reserved.
###decoder MTL
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_transformer-decoder_E2_L10 > test_log_focus/test_DECODER_MTL.log &
#CUDA_VISIBLE_DEVICES=6 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_transformer-decoder_E2_L10 > test_log_focus/test_DECODER_MTL_ppl.log &

###encdec MTL
#CUDA_VISIBLE_DEVICES=1 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_transformer-encdec_E2_L10 > test_log_focus/test_ENCDEC_MTL.log &
#CUDA_VISIBLE_DEVICES=6 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_transformer-encdec_E2_L10 > test_log_focus/test_ENCDEC_MTL_ppl.log &

###gpt2 MTL
#CUDA_VISIBLE_DEVICES=2 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10 > test_log_focus/test_GPT_MTL.log &
#CUDA_VISIBLE_DEVICES=6 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10 > test_log_focus/test_GPT_MTL_ppl.log &

###bart MTL
#CUDA_VISIBLE_DEVICES=3 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10 > test_log_focus/test_BART_MTL.log &
#CUDA_VISIBLE_DEVICES=6 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10 > test_log_focus/test_BART_MTL_ppl.log &

###########ablations##########
###gpt2 LM
#CUDA_VISIBLE_DEVICES=7 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_LM > test_log_focus/test_GPT_LM.log &
#CUDA_VISIBLE_DEVICES=0 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_LM > test_log_focus/test_GPT_LM_ppl.log &

###bart LM
#CUDA_VISIBLE_DEVICES=7 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_LM > test_log_focus/test_BART_LM.log &
#CUDA_VISIBLE_DEVICES=1 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_LM > test_log_focus/test_BART_LM_ppl.log &


##gpt2 WOKS
#CUDA_VISIBLE_DEVICES=2 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_KS > test_log_focus/test_GPT_wo_ks.log &
#CUDA_VISIBLE_DEVICES=2 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_KS > test_log_focus/test_GPT_wo_ks_ppl.log &

#
##bart WOKS
#CUDA_VISIBLE_DEVICES=3 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_KS > test_log_focus/test_BART_wo_ks.log &
#CUDA_VISIBLE_DEVICES=3 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_KS > test_log_focus/test_BART_wo_ks_ppl.log &


##gpt2 WOPS
#CUDA_VISIBLE_DEVICES=6 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_PS > test_log_focus/test_GPT_wo_ps.log &
#CUDA_VISIBLE_DEVICES=6 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_WO_PS > test_log_focus/test_GPT_wo_ps_ppl.log &


##bart WOPS
#CUDA_VISIBLE_DEVICES=7 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_PS > test_log_focus/test_BART_wo_ps.log &
#CUDA_VISIBLE_DEVICES=7 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_WO_PS > test_log_focus/test_BART_wo_ps_ppl.log &


##########large models##########
##gpt2 medium
#CUDA_VISIBLE_DEVICES=3 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_large > test_log_focus/test_GPT2_large_MTL.log &
#CUDA_VISIBLE_DEVICES=3 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_large > test_log_focus/test_GPT2_large_MTL_ppl.log &

##bart large
#CUDA_VISIBLE_DEVICES=4 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_large > test_log/test_BART_large_MTL.log &
#CUDA_VISIBLE_DEVICES=4 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_large > test_log/test_BART_large_MTL_ppl.log &

##gpt2 medium LM
#CUDA_VISIBLE_DEVICES=5 nohup python evaluate_test.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_large_LM > test_log/test_GPT2_large_LM.log &
#CUDA_VISIBLE_DEVICES=5 nohup python evaluate_test_ppl.py --model_name GPT2 --model_checkpoint /home/mnt/user/focus/train_focus_GPT2_E2_L10_large_LM > test_log/test_GPT2_large_LM_ppl.log &

##bart large LM
#CUDA_VISIBLE_DEVICES=6 nohup python evaluate_test.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_large_LM > test_log/test_BART_large_LM.log &
#CUDA_VISIBLE_DEVICES=6 nohup python evaluate_test_ppl.py --model_name BART --model_checkpoint /home/mnt/user/focus/train_focus_BART_E2_L10_large_LM > test_log/test_BART_large_LM_ppl.log &

echo
