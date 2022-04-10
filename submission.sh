#!/bin/bash

submission_path=/home/energy/amawi/projects/mol-td/run.sh

cmd = "--wb \
       -p test \
       -m HierarchicalTDVAE \
       -nt 10 \
       "

sbatch --gres=gpu:RTX3090 --job-name=actsweep $submission_path $cmd


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


