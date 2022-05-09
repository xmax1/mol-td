import re
import os
import itertools

# parser.add_argument('-m', '--model', default='HierarchicalTDVAE', type=str)
# parser.add_argument('-t', '--transfer_fn', default='GRU', type=str)
# parser.add_argument('-enc', '--encoder', default='GNN', type=str)  # GCN for graph, MLP for line
# parser.add_argument('-dec', '--decoder', default='MLP', type=str)  # GCN for graph, MLP for line
# parser.add_argument('-nt', '--n_timesteps', default=8, type=int)
# parser.add_argument('-net', '--n_eval_timesteps', default=8, type=int)
# parser.add_argument('-new', '--n_eval_warmup', default=None, type=int)

# parser.add_argument('-nenc', '--n_enc_layers', default=2, type=int)
# parser.add_argument('-ndec', '--n_dec_layers', default=2, type=int)
# parser.add_argument('-tl', '--n_transfer_layers', default=2, type=int)
# parser.add_argument('-ne', '--n_embed', default=40, type=int)
# parser.add_argument('-nl', '--n_latent', default=2, type=int)
# parser.add_argument('-ystd', '--y_std', default=0.01, type=float)
# parser.add_argument('-b', '--beta', default=1., type=float)
# parser.add_argument('-lp', '--likelihood_prior', default=False, type=input_bool)
# parser.add_argument('-cw', '--clockwork', default=False, type=input_bool)
# parser.add_argument('-mj', '--mean_trajectory', default=False, type=input_bool)

# parser.add_argument('-e', '--n_epochs', default=50, type=int)
# parser.add_argument('-bs', '--batch_size', default=128, type=int)
# parser.add_argument('-lr', '--lr', default=0.001, type=float)


# hyperparameters1 = {
#     '-t': ['LSTM', ],  # LSTM
#     '-nenc': [1, ],
#     '-ne': [40, ],
#     '-nl': [1, 2, 3],
#     '-cw': ['True', 'False'],  # clockwork
#     '-mj': ['True',],  # mean trajectory
#     '-ystd': [0.05, ],
#     '-bs': [128, ],
#     '-lag': [3, ],
#     '-e': [30, ],  # epochs

#     '-d': ['md17/uracil_dft', 'md17/ethanol_dft', 'md17/malonaldehyde_dft', 'md17/naphthalene_dft',
#             'md17/aspirin_dft.npz', 'md17/salicylic_dft.npz', 'md17/toluene_dft.npz', 'md17/benzene2017_dft.npz'],
    
#     '-tag': ['monday_clockwork', ],
# }

hyperparameters1 = {
    '-nl': [1, 2, 3],
    '-cw': ['True', 'False'],  # clockwork
    '-mj': ['True',],  # mean trajectory
    '-ystd': [0.01, ],
    '-bs': [128, ],
    '-e': [30, ],  # epochs

    '-d': ['md17/uracil_dft', 'md17/ethanol_dft', 'md17/malonaldehyde_dft', 'md17/naphthalene_dft',
            'md17/aspirin_dft.npz', 'md17/salicylic_dft.npz', 'md17/toluene_dft.npz', 'md17/benzene2017_dft.npz'],
    
    '-tag': ['smaller_latent', ],
}

addendum = ' --wb -p TimeDynamics_v2 -g smaller_latent'

lists = [[f' {k} {str(v)}' for v in options] for k, options in hyperparameters1.items()]

cmds = [p for p in itertools.product(*lists)]
cmds = [" ".join(l) + addendum for l in cmds]

run='./run.sh'

with open('experiments.txt', 'w') as f:
    for cmd in cmds:
        f.write(cmd)
        f.write('\n')

print(f'{len(cmds)} experiments ready to go, view at experiments.txt \n Top 10:')
for l in cmds[:10]:
    print(l)
print(f'Number of experiments: {len(cmds)}')

user_input = input("y for launch, n for no no no, c for complete list then another request \n")

if user_input == 'c':
    for l in cmds:
        print(l)
    user_input = input("y for launch, n for no no no \n")

if user_input == 'y':
    os.system(f'sbatch --array=0-{len(cmds)-1}%20  ./run.sh')


# cmds = []
# for t in ts:
#     for nt in nts:
#         for ed in eds: 
#             for tl in tls:
#                 for ne in nes:
#                     for nl in nls:
#                         for y_std in y_stds:
#                             for beta in betas:
#                                 for sc in scs:
#                                     for bs in bss:
#                                         cmd = f'--wb \
#                                                 -t {t} \
#                                                 -nt {nt} \
#                                                 -el {ed} \
#                                                 -dl {ed} \
#                                                 -tl {tl} \
#                                                 -ne {ne} \
#                                                 -nl {nl} \
#                                                 -y_std {y_std} \
#                                                 -b {beta} \
#                                                 -e 10 \
#                                                 -lr 0.001 \
#                                                 {sc} \
#                                                 -bs {bs} \
#                                                 -g prior_into_posterior \
#                                                 -p TimeDynamics \
#                                                 -tag {tag}'
#                                         cmds.append(re.sub(r'\s+', ' ', cmd))

# # import re
# # test = '     Good    Sh ow     '
# # re.sub(r'\s+', ' ', test)



