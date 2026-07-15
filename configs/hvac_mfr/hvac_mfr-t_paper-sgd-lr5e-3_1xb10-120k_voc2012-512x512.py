_base_ = './hvac_mfr-t_paper-sgd-lr1e-3_1xb10-120k_voc2012-512x512.py'

optim_wrapper = dict(
    optimizer=dict(lr=0.005))
