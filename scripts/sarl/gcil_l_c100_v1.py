import os

best_params = {
    200: {
        'idt': 'v1',
        'reg_weight': 0.5,
        'kw': '0.9 0.9 0.9 0.9',
        'lr': 0.03,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'lr_steps': '35 45'
    },
    500: {
        'idt': 'v1',
        'reg_weight': 0.5,
        'kw': '0.9 0.9 0.9 0.9',
        'lr': 0.03,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'lr_steps': '35 45'
      },
    5120: {
        'idt': 'v1',
        'reg_weight': 0.50,
        'kw': '0.9 0.9 0.9 0.9',
        'lr': 0.03,
        'minibatch_size': 32,
        'batch_size': 32,
        'n_epochs': 50,
        'lr_steps': '35 45'
    },
}


lst_kw = [0.7]
lst_apply_kw = ["1 1 1 1"]
lst_kw_local = [1]
lst_reg = [0.2]
lst_fwd_reg = [1]
lst_buff_logit_weight = [0]

lst_cp_weight = [0.5, 1]
lst_cont_weight = [0.1, 0.01]

lst_buffer_update = [0]
lst_buffer_pass = [1]
lst_dist_thresh = [0.8]

lst_seed = [0, 1, 2]
lst_buffer_size = [200, 500]
count = 0

for seed in lst_seed:
    for buffer_size in lst_buffer_size:
        # Objective weights
        for cp_weight in lst_cp_weight:
            for cont_weight in lst_cont_weight:
                for reg_weight in lst_reg:
                    for fwd_reg_weight in lst_fwd_reg:
                        for apply_kw in lst_apply_kw:
                            for kw in lst_kw:
                                for kw_local in lst_kw_local:
                                    params = best_params[buffer_size]
                                    exp_id = f"SSSL-gcil-l-c100-kw-{buffer_size}-kw-{''.join(apply_kw.split())}-{kw}-{kw_local}-w--{cp_weight}-{cont_weight}-{reg_weight}-{fwd_reg_weight}-{lst_dist_thresh[0]}-s-{seed}"
                                    job_args = f"python main.py  \
                                        --experiment_id {exp_id} \
                                        --model sarl_single_ser_local \
                                        --fwd_reg_weight {fwd_reg_weight} \
                                        --dataset gcil-cifar100 \
                                        --weight_dist longtail \
                                        --apply_kw {apply_kw} \
                                        --kw_local {kw_local} \
                                        --kw {kw} {kw} {kw} {kw} \
                                        --cp_weight {cp_weight} \
                                        --cont_weight {cont_weight} \
                                        --reg_weight {reg_weight} \
                                        --buff_ce_weight 1 \
                                        --buffer_update 0 \
                                        --dist_thresh {lst_dist_thresh[0]} \
                                        --buffer_pass {lst_buffer_pass[0]} \
                                        --buffer_size {buffer_size} \
                                        --batch_size {params['batch_size']} \
                                        --minibatch_size {params['minibatch_size']} \
                                        --lr {params['lr']} \
                                        --use_lr_scheduler 1 \
                                        --lr_steps {params['lr_steps']} \
                                        --n_epochs {params['n_epochs']} \
                                        --output_dir experiments/sarl_ser_single_local_gcil \
                                        --csv_log \
                                        --seed {seed} \
                                        --use_supercat 0 \
                                        --device cuda:6 \
                                        --save_model 0 \
                                        --save_interim 0 \
                                        "
                                    count += 1
                                    os.system(job_args)

print('%s jobs counted' % count)
