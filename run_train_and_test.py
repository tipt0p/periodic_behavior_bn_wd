#! /usr/bin/env python3
import os
import subprocess
import argparse
import datetime
import time

parser = argparse.ArgumentParser(description='Managing experiments')
parser.add_argument('--test', action='store_true',
                        help='print (test) or os.system (run)')

args = parser.parse_args()
    
if args.test:
    action = print
else:
    action = os.system

ENVIRONMENT = 'CycleBNNet_env'
 
data = 'CIFAR10'#'CIFAR10''CIFAR100'
network = 'ConvNetSI'#'ResNet18SI''ConvNetSI''ResNet18SIAf''ConvNetSIAf'
fix_noninvlr = 0.0

save_path = './Experiments/{}_{}/'.format(network,data)
if fix_noninvlr >=0:
    save_path = './Experiments/{}_{}_noninvlr_{}/'.format(network,data,fix_noninvlr) 
if not os.path.exists('./Experiments'):
    os.mkdir('./Experiments')
if not os.path.exists(save_path):
    os.mkdir(save_path)

params = {'dataset' : data,
          'model': network,
          'noninvlr':fix_noninvlr,
          'momentum': 0.0,
          'num_channels':32,
          'depth':3,# work only for ConvNet
          'epochs': 1001,
          'corrupt_train': 0.0,
          'save_freq': 1,
          'eval_freq':1000,
          'use_data_size':50000,
          'dir': save_path + 'checkpoints',
          'init_scale':10.,
          'fix_si_pnorm_value':-1,
          'gpu':0
         }

lrs = [0.01,]    
wds = [0.001,]

add_params = '--use_test --no_schedule --no_aug'#--fbgd --fix_si_pnorm

params_test = {'dataset' : data,
          'model': network,
          'num_channels': params['num_channels'],
          'depth': params['depth'],
          'init_scale':params['init_scale'],
          'save_path': save_path + 'info',
          'models_dir': save_path + 'checkpoints',
          'use_data_size':params['use_data_size'],
          'gpu':params['gpu']
         }

log_path = save_path + 'logs/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(log_path):
    os.mkdir(log_path)
    
info_path = save_path + 'info/'
if not os.path.exists(info_path):
    os.mkdir(info_path)

commands = []

for ind in range(len(lrs)):
    p = params.copy()
    p['lr_init'] = lrs[ind]
    p['wd'] = wds[ind]

    p_test = params_test.copy()

    exp_name = 'c{}_d{}_ds{}_lr{}_wd{}_mom{}_corr{}_epoch{}'.format(p['num_channels'],p['depth'],p['use_data_size'],p['lr_init'],p['wd'],p['momentum'],p['corrupt_train'],p['epochs'])
    if 'no_schedule' in add_params:
        exp_name = exp_name + '_nosch'
    if p['init_scale'] >0:
        exp_name = exp_name + 'initscale{}'.format(p['init_scale'])
    if 'no_aug' in add_params:
        exp_name = exp_name + '_noaug'
    if 'fbgd' in add_params:
        exp_name = exp_name + '_fbgd'
    if 'fix_si_pnorm' in add_params:
        exp_name = exp_name + '_fix_si_pnorm{}'.format(p['fix_si_pnorm_value'])
        

    p['dir'] = params['dir'] + '/' + exp_name
    exp_log_path = log_path + exp_name

    p_test['models_dir'] = params_test['models_dir'] + '/' + exp_name + '/trial_0'

    # train
    command = 'train.py {} >> {}'.format(' '.join(["--{} {}".format(k,v) for (k, v) in p.items()])+' ' +add_params, exp_log_path+'.out')
    commands.append(command)

    #train metrics
    p_test['save_path'] = params_test['save_path'] + '/' + exp_name + '/train-tm.npz'
    commands.append('get_info.py {} --corrupt_train {} --train_mode --eval_model --all_pnorm'.format(' '.join(["--{} {}".format(k,v) for (k, v) in p_test.items()]), p['corrupt_train']))
    commands.append('get_info.py {} --corrupt_train {} --train_mode --update  --calc_grad_norms'.format(' '.join(["--{} {}".format(k,v) for (k, v) in p_test.items()]), p['corrupt_train']))

    #test metrics
    p_test['save_path'] = params_test['save_path'] + '/' + exp_name + '/test-em.npz'
    commands.append('get_info.py {} --use_test --eval_model'.format(' '.join(["--{} {}".format(k,v) for (k, v) in p_test.items()])))
           
                    
if ENVIRONMENT:
    tmp_str = ' && ~/anaconda3/envs/{}/bin/python '.format(ENVIRONMENT)
    final_command = "bash -c '. activate {} {} {}'".format(ENVIRONMENT,tmp_str,tmp_str.join(commands))
else:
    final_command = 'python '.join(command)

action(final_command)
                  