# test scalings first for 20 poisons. 
# bigger beta ==> more importance to feature distance, less importance to image space.
# smaller beta ==> less important to feature distance, more importance to image space.

# python poisoning.py --dataset mnist --method softmax --ckpt_path ../experiments/softmax_mnist/models/epoch-best.model --base_strategy random --seed 176 --max_poisons 20 --beta 0.00083 --poison_lr 38250000 --normalize_feats --loss_thres 1e-4 --exp_name beta_0_00083
# python poisoning.py --dataset mnist --method softmax --ckpt_path ../experiments/softmax_mnist/models/epoch-best.model --base_strategy random --seed 176 --max_poisons 20 --beta 0.0083 --poison_lr 38250000 --normalize_feats --loss_thres 1e-4 --exp_name beta_0_0083
# python poisoning.py --dataset mnist --method softmax --ckpt_path ../experiments/softmax_mnist/models/epoch-best.model --base_strategy random --seed 176 --max_poisons 20 --beta 0.083 --poison_lr 38250000 --normalize_feats --loss_thres 1e-4 --exp_name beta_0_083

# 0.0083 works best, turns out.

# GENERATING 100 POISONS FOR SOFTMAX
# python poisoning.py --dataset mnist --method softmax --ckpt_path ../experiments/softmax_mnist/models/epoch-best.model --base_strategy random --seed 176 --max_poisons 1000 --beta 0.0083 --poison_lr 38250000 --normalize_feats --loss_thres 1e-4 --exp_name mnist_softmax_poisons

# GENERATING 1000 POISONS FOR LGM
# python poisoning.py --dataset mnist --method lgm --ckpt_path ../experiments/lgm_mnist/lgm-model --base_strategy random --seed 176 --max_poisons 1000 --beta 0.0083 --poison_lr 38250000 --normalize_feats --loss_thres 1e-4 --exp_name mnist_lgm_poisons

# GENERATING POISONS FOR CIFAR10 NOW.
# python poisoning.py --dataset cifar10 --method lgm --ckpt_path ../checkpoints/LGM-cifar-vgg/LGM-vgg-cifar.epoch-10-.model --base_strategy random --seed 176 --max_poisons 100 --beta 0.0083 --poison_lr 38250000 --normalize_feats --loss_thres 1e-4 --exp_name cifar10_lgm_poisons
