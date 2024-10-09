#!/bin/bash

#SBATCH --job-name=CSKD
#SBATCH --nodes=1
#SBATCH --partition=himem
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

#SBATCH --output test-%j.txt
#SBATCH --error test-%j.err

# gpu -> gpu ((a40) or rtx --> gpu:a40:1) or himem


# ==============
# train teacher:
# python main_teacher.py --arch=wide_resnet-40-2

# teacher inference:
# python main_teacher.py --mode=eval --arch=wide_resnet-40-2


# train student:
#python main_student.py --teacherArch=wide_resnet-40-2 --studentTask=subclass-cifar10 \
#       --studentArch=wide_resnet-16-1 --teacher_model_name=teacher-cifar10-wide_resnet-40-2-checkpoint \
#       --teacherClass=10 --studentClass=5 --student_model_name=student-cifar10-wide_resnet-16-1 \
#       --renormHyper=0.1 --icvHyper=0.1 --oplHyper=0.1


# student inference:
python main_student.py --mode=eval --studentTask=subclass-cifar10 \
       --studentArch=wide_resnet-16-1 --teacherClass=10 --studentClass=5 \
       --student_model_name=student-cifar10-wide_resnet-16-1-checkpoint





