from templates import *
from templates_latent import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_samples','--es', type=int, default=50_00)
    parser.add_argument('--steps','--s', type=int, default=20)
    args = parser.parse_args()
    # train the autoenc moodel
    # this requires V100s.
    #gpus = [0]
    #conf = ffhq128_autoenc_130M()
    #conf.batch_size = 25
    #train(conf, gpus=gpus)
#
    ## infer the latents for training the latent DPM
    ## NOTE: not gpu heavy, but more gpus can be of use!
    #gpus = [0, 1, 2, 3]
    #conf.eval_programs = ['infer']
    #train(conf, gpus=gpus, mode='eval')
#
    ## train the latent DPM
    ## NOTE: only need a single gpu
    gpus = [0]
    conf = ffhq128_autoenc_latent()
    ##train(conf, gpus=gpus)
#
    ## unconditional sampling score
    ## NOTE: a lot of gpus can speed up this process
    ##gpus = [0, 1, 2, 3]
    #conf.use_fixed_groupnorm = False
    #conf.fixed_groupnorm_mode = 'per_layer'
    #conf.groupnorm_strategy = 'per_group'
    conf.eval_programs = [f'fid({args.steps},{args.steps})']
    train(conf, gpus=gpus, mode='eval', eval_samples=args.eval_samples)