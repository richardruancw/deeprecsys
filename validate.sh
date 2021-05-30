# Test attention implementation, this setting == train attention model directly
run_reweight_v2.py --model seq --cuda_idx 0 --lambda_ 0 --w_lower_bound 1.0 --g_step 0 --w_step 0
#expect relevance score
#INFO:deeprecsys.eval:Full empirical relevance score: 0.015552980132450437


# Test the reweight training using attention
run_reweight_v2.py --model seq --cuda_idx 0 --lambda_ 1 --w_lower_bound 1.0 --g_step 1 --w_step 1
#INFO:root:biased eval for SVD model on test
#INFO:deeprecsys.eval:Full empirical relevance score: 0.01585596026490078
#INFO:root:------Reweight and rebalance model ------
#DEBUG:deeprecsys.data:Build windowed data
#DEBUG:deeprecsys.data:Build windowed data
#DEBUG:deeprecsys.data:Build windowed data
#4825
#INFO:deeprecsys.eval:Full empirical relevance score: 0.015360927152317965
#4825
#INFO:deeprecsys.eval:Full empirical relevance score: 0.01661589403973525
#4825
#INFO:deeprecsys.eval:Full empirical relevance score: 0.016677152317880944