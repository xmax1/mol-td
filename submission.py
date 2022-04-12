import re
import os
import itertools

ts=('-t GRU', )
nts=('-nt 5', )
els=('-el 1 -dl 1', )
# dls=(x.replace('e', 'd') for x in els)
tls=('-tl 1', )
nes=('-ne 30', )
nls=('-nl 1', '-nl 2', '-nl 3')
y_stds=('-y_std 0.01', '-y_std 0.05', '-y_std 0.1', '-y_std 0.5', '-y_std 1.')
betas=('-b 1','-b 10','-b 100', '-b 1000')
post_into_prior=('--post_into_prior', '')
scs=('', )
lps=('--likelihood_prior', '')
bss=('-bs 128',)
lrs=('-lr 0.001', )
es=('-e 10', )
tag=('-tag no_tag', )
other=('--wb -p TimeDynamics -g initial_sweep4', )

list = [ts, nts, els, tls, nes, nls, y_stds, betas, post_into_prior, scs, bss, lrs, es, tag, other]
cmds = [p for p in itertools.product(*list)]
cmds = [" ".join(l) for l in cmds]

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



