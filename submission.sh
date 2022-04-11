#!/bin/bash

submission_path=/home/energy/amawi/projects/mol-td/run.sh

arr=()
while IFS= read -r line; do
  arr+=("$line")
done < file

while IFS= read -r line; do   arr+=("$line"); done < experiments.txt

ts=(LSTM GRU)
nts=(5 10)
eds=(1 2)
tls=(1 2)
nes=(20 40)
nls=(1 2 3)
y_stds=(0.5 1.0)
betas=(1 2000 4000 8000)
scs=(--skip_connections "")
bss=(32 128)

i=0

for t in "${ts[@]}"
do 
  for nt in "${nts[@]}"
  do
    for ed in "${eds[@]}"
    do
      for tl in "${tls[@]}"
      do
        for ne in "${nes[@]}"
        do
          for y_std in "${y_stds[@]}"
          do
            for beta in "${betas[@]}"
            do
              for sc in "${scs[@]}"
              do
                for bs in "${bss[@]}"
                do
                  cmd="--wb \
                     -t $t \
                     -m HierarchicalTDVAE \
                     -nt $nt \
                     -el $ed \
                     -dl $ed \
                     -tl $tl \
                     -ne $ne \
                     -y_std $y_std \
                     -b $beta \
                     -bs $bs $sc \
                     -g initial_sweep \
                     -p TimeDynamics"
                  i=$((i+1))
                  sbatch $submission_path $cmd
                  echo $i
                  echo $cmd
                  exit
                done
              done
            done
          done
        done
      done
    done
  done
done




# cmd="--wb \
#        -p test \
#        -m HierarchicalTDVAE \
#        -nt 10 \
#        -i niflheim
#        "

# sbatch --gres=gpu:RTX3090 --job-name=actsweep $submission_path $cmd


# if [[ $2 -eq 0 ]]; then
#     ngpu=1
# else
#     ngpu=$2
# fi


# # for i in "${!myArray[@]}"
# #     # for hypam in "${myArray[@]}"
# #     do
# #         hypam=${myArray[i]}

# if [ "$1" == "twoloop" ]  # the space is needed because [ is a test ]
# then
#     echo Calling twoloop submission

#     myArray=(cos 2cos 3cos 4cos 2cos+2sin 3cos+3sin)
#     myArray2=(tanh cos)
#     # myArray=(16 32 64 128)
#     # myArray2=(4 8 16 32)
#     for hypam in "${myArray[@]}"
#     do
#         for hypam2 in "${myArray2[@]}"
#         do
#             cmd = "$hypam2"
#             sbatch --gres=gpu:RTX3090:$ngpu --job-name=actsweep $submission_path $cmd
#             echo $hypam $hypam2
#             sleep 20
#         done
#     done
# fi


