
set -ex
 CUDA_VISIBLE_DEVICES=0 python adversarial_visible_watermarking/main.py  --epochs 1\
 --schedule 1\
 --lr 1e-4\
 --resume adversarial_visible_watermarking/wm_removers/ckpt/SLBR.pth.tar\
 --arch slbr\
 -c /path/to/your/save_path\
 --attack_method pgd_inn\
 --limited-dataset 1\
 --use_rie True\
 --epsilon 4\
 --step_alpha 1\
 --stopnum 5\
 --iters 50\
 --rie_iters 200\
 --simage True\
 --lambda_p 200\
 --input-size 256\
 --train-batch 1\
 --test-batch 1\
 --use_refine \
 --base-dir /path/to/your/dataset\
 --data dataset_name