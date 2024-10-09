import sys, os
import numpy as np
import torch
from models import resnet
from models import wide_resnet


def adjust_learning_rate(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7, logger=None):
    """Decay learning rate by a factor of 0.2 every lr_decay_epoch epochs."""
    steps = np.sum(epoch > np.asarray(lr_decay_epoch))
    if steps > 0:
        lr = init_lr * (0.2 ** steps)
        logger.add_line('Learning rate:            {}'.format(lr))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    return optimizer


def accuracy(predictions, targets, not_class=5, axis=1):
    batch_size = predictions.size(0)
    predictions = predictions.max(axis)[1].type_as(targets)

    hits = predictions.eq(targets)
    acc = 100. * hits.sum().float() / float(batch_size)
    return acc


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(model_dir, state, ignore_tensors=None, teacher_or_student='teacher'):
    checkpoint_fn = os.path.join(model_dir, teacher_or_student + '-checkpoint.pth.tar')
    if ignore_tensors is not None:
        for p in ignore_tensors.values():
            if p in state['state_dict']:
                del state['state_dict'][p]
    torch.save(state, checkpoint_fn)


class Logger(object):
    def __init__(self, log2file=False, mode='train', model_dir=None, teacher_student='teacher'):
        if log2file:
            assert model_dir is not None
            fn = os.path.join(model_dir, '{}-{}.log'.format(teacher_student, mode))
            self.fp = open(fn, 'w')
        else:
            self.fp = sys.stdout

    def add_line(self, content):
        self.fp.write(content+'\n')
        self.fp.flush()


def get_tables(data_set_name, total_labels, num_interested_labels, category=None):
    
    seed = 0
    rng = np.random.default_rng(seed)
    random_class_labels = rng.permutation(total_labels)[:num_interested_labels]
    trueLabel2fakeLabel = fakeLabel2trueLabel = None
    if data_set_name == 'cifar100':
        trueLabel2fakeLabel = {random_class_labels[idx]: idx for idx in range(num_interested_labels)}
        fakeLabel2trueLabel = {idx: random_class_labels[idx] for idx in range(num_interested_labels)}
    elif data_set_name == 'tinyimagenet':
        trueLabel2fakeLabel = {random_class_labels[idx]: idx for idx in range(num_interested_labels)}
        fakeLabel2trueLabel = {idx: random_class_labels[idx] for idx in range(num_interested_labels)}
    elif data_set_name == 'cifar10':
        trueLabel2fakeLabel = {random_class_labels[idx]: idx for idx in range(num_interested_labels)}
        fakeLabel2trueLabel = {idx: random_class_labels[idx] for idx in range(num_interested_labels)}
    elif data_set_name == 'cub2002011':
        trueLabel2fakeLabel = {random_class_labels[idx]: idx for idx in range(num_interested_labels)}
        fakeLabel2trueLabel = {idx: random_class_labels[idx] for idx in range(num_interested_labels)}
    elif data_set_name == 'mit67':
        trueLabel2fakeLabel = {random_class_labels[idx]: idx for idx in range(num_interested_labels)}
        fakeLabel2trueLabel = {idx: random_class_labels[idx] for idx in range(num_interested_labels)}

    return trueLabel2fakeLabel, fakeLabel2trueLabel


def get_model(input_arch_name, num_classes, pretrained=None):

    if input_arch_name == 'resnet18': 
        return resnet.create_model('resnet18', pretrained=pretrained, num_classes=num_classes)
    elif input_arch_name == 'resnet34': 
        return resnet.create_model('resnet34', pretrained=pretrained, num_classes=num_classes)
    elif input_arch_name == 'resnet50': 
        return resnet.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
    elif input_arch_name == 'resnet101': 
        return resnet.create_model('resnet101', pretrained=pretrained, num_classes=num_classes)
    elif input_arch_name == 'resnet152': 
        return resnet.create_model('resnet152', pretrained=pretrained, num_classes=num_classes)
    elif input_arch_name == 'wide_resnet-40-2': 
        return wide_resnet.Wide_ResNet(40, 2, 0.05, num_classes)
    elif input_arch_name == 'wide_resnet-16-1': 
        return wide_resnet.Wide_ResNet(16, 1, 0.05, num_classes)
    else:
        raise Exception('Undefined Model!')

