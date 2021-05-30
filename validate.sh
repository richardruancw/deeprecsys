# Test attention implementation, this setting == train attention model directly
python run_reweight_v2.py --model seq --cuda_idx 0 --lambda_ 0 --w_lower_bound 1.0 --g_step 0 --w_step 0
#expect relevance score
#INFO:deeprecsys.eval:Full empirical relevance score: 0.015552980132450437

# Test the reweight training using attention with bound
python run_reweight_v2.py --model seq --cuda_idx 0 --lambda_ 1 --w_lower_bound 1.0 --g_step 1 --w_step 1
#INFO:deeprecsys.eval:Full empirical relevance score: 0.016677152317880944

# Use different baseclass for w and g
python run_reweight_v2.py --model seq --cuda_idx 0 --lambda_ 1 --w_lower_bound 0.1 --g_step 1 --w_step 1 --w_model mlp --g_model mlp

# Use attention model for g and w shared item embedding from f
python run_reweight_v2.py --model seq --cuda_idx 0 --lambda_ 1 --w_lower_bound 0.1 --g_step 1 --w_step 1 --share_f_embed

# Use mf model for g and w shared item embedding from f
python run_reweight_v2.py --model mf --cuda_idx 0 --lambda_ 1 --w_lower_bound 0.1 --g_step 1 --w_step 1 --share_f_embed