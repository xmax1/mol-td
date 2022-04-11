import re
import os

ts=['LSTM',]
nts=(5,)
eds=(2,)
tls=(2,)
nes=(30,)
nls=(1, 2, 3)
y_stds=(1.0,)
betas=(1, 2000, 8000)
scs=('--skip_connections', '')
bss=(32, 128, 256)

run='./run.sh'

cmds = []
for t in ts:
    for nt in nts:
        for ed in eds: 
            for tl in tls:
                for ne in nes:
                    for nl in nls:
                        for y_std in y_stds:
                            for beta in betas:
                                for sc in scs:
                                    for bs in bss:
                                        cmd = f'--wb \
                                                -t {t} \
                                                -nt {nt} \
                                                -el {ed} \
                                                -dl {ed} \
                                                -tl {tl} \
                                                -ne {ne} \
                                                -nl {nl} \
                                                -y_std {y_std} \
                                                -b {beta} \
                                                {sc} \
                                                -bs {bs} \
                                                -g initial_sweep2 \
                                                -p TimeDynamics'
                                        cmds.append(re.sub(r'\s+', ' ', cmd))

# import re
# test = '     Good    Sh ow     '
# re.sub(r'\s+', ' ', test)

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
