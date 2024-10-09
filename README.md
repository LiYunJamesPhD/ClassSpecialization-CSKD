# Class Specialized Knowledge Distillation (Accepted to Asian Conference on Computer Vision 2022)

Paper: [Link](https://openaccess.thecvf.com/content/ACCV2022/papers/Wang_Class_Specialized_Knowledge_Distillation_ACCV_2022_paper.pdf)

## Abstract
Knowledge Distillation (KD) is a compression framework that transfers distilled knowledge from a teacher to a smaller student model. KD approaches conventionally address problem domains where the teacher and student network have equal numbers of classes for clas- sification. We provide a knowledge distillation solution tailored for class specialization, where the user requires a compact and performant net- work specializing in a subset of classes from the class set used to train the teacher model. To this end, we introduce a novel knowledge dis- tillation framework, Class Specialized Knowledge Distillation (CSKD), that combines two loss functions: Renormalized Knowledge Distillation (RKD) and Intra-Class Variance (ICV) to render a computationally- efficient, specialized student network. We report results on several pop- ular architectural benchmarks and tasks. In particular, CSKD consis- tently demonstrates significant performance improvements over teacher models for highly restrictive specialization tasks (e.g., instances where the number of subclasses or datasets is relatively small), in addition to outperforming other state-of-the-art knowledge distillation approaches for class specialization tasks.

## Usage
### Requirements
```
1. Python>=3.8
2. PyTorch>=1.7
3. Numpy
```

### Training
To train a teacher model and obtain the trained model, run
```
python main_teacher.py --arch=wide_resnet-40-2
```

To train a student model and obtain a compressed model, run
```
python main_student.py --teacherArch=wide_resnet-40-2 --studentTask=subclass-cifar10 \
       --studentArch=wide_resnet-16-1 --teacher_model_name=teacher-cifar10-wide_resnet-40-2-checkpoint \
       --teacherClass=10 --studentClass=5 --student_model_name=student-cifar10-wide_resnet-16-1 \
       --renormHyper=0.1 --icvHyper=0.1 --oplHyper=0.1
```

### Inference
To test the trained teacher model, run
```
python main_teacher.py --mode=eval --arch=wide_resnet-40-2
```

To test the trained student model (compressed model), run
```
python main_student.py --mode=eval --studentTask=subclass-cifar10 \
       --studentArch=wide_resnet-16-1 --teacherClass=10 --studentClass=5 \
       --student_model_name=student-cifar10-wide_resnet-16-1-checkpoint
```

### Citation
If you feel our work is useful, please cite our work:
```
@inproceedings{wang2022class,
  title={Class specialized knowledge distillation},
  author={Wang, Li-Yun and Rhodes, Anthony and Feng, Wu-chi},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={247--264},
  year={2022}
}
```


