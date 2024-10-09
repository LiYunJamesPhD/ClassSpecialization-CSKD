import math
import os
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import dataloaders
import utils
from losses import *


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--teacherArch', default='resnet34', help='teacher architecture (default: resnet34)')
    parser.add_argument('--studentTask', default='subclass-cifar10', help='task to train (student)')
    parser.add_argument('--studentArch', default='resnet18', help='student architecture (default: resnet18)')

    parser.add_argument('--mode', default='train', help='mode to training or evaluation.')
    parser.add_argument('--model_dir', default='./checkpoints', help='model directory')
    parser.add_argument('--student_model_name', default='', help='model name to a student network')
    parser.add_argument('--teacher_model_name', default='', help='model name to a pre-trained teacher network')
    parser.add_argument('--teacherClass', type=int, default=10, help='number of classes (teacher network)')
    parser.add_argument('--studentClass', type=int, default=10, help='number of classes (student  network)')
    parser.add_argument('--pretrainedStudent', action='store_true', help='download a pre-trained student model')

    parser.add_argument('--log2file', action='store_true', help='log output to file (under model_dir/train.log)')

    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr-decay-epochs', default="60,120,160", help='number of epochs for each lr decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=0.0005, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--CEHyper', default=1.0, type=float, help='hyperparameter to cross entropy loss (default: 1.0)')
    parser.add_argument('--renormHyper', default=0.2, type=float, help='hyperparameter to renormalized loss (default: 0.2)')
    parser.add_argument('--icvHyper', default=0.2, type=float, help='hyperparameter to intra-class variation loss (default: 0.2)')
    parser.add_argument('--oplHyper', default=0.2, type=float, help='hyperparameter to OPL (default: 0.2)')

    parser.add_argument('--print-freq', default=100, type=int, help='print frequency (default: 10 iter)')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    return parser.parse_args()


def train(student_train_loader, teacher_model, student_model, student_criterion,
          student_optimizer, e, DEVICE, opt, renormalized_kd_loss, opl, icv):
    
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    loss_avg = utils.AverageMeter()
    acc_avg = utils.AverageMeter()
    kd_loss_1_avg = utils.AverageMeter()
    kd_loss_3_avg = utils.AverageMeter()
    opl_avg = utils.AverageMeter()

    teacher_model.eval()
    student_model.train()
    
    end = time.time()
    for i, (imgs, labels) in enumerate(student_train_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, labels = Variable(imgs), Variable(labels)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        student_logit, student_feats = student_model(imgs)  # check student model
        student_cross_loss = student_criterion(student_logit, labels)
        
        loss_avg.update(student_cross_loss.item(), imgs.size(0))
        
        acc = utils.accuracy(student_logit, labels)
        acc_avg.update(acc.item(), imgs.size(0))

        # get teacher's logits
        with torch.no_grad():
            output_teacher_batch, _ = teacher_model(imgs)

        # Renormalized knowledge distillation loss:
        renormal_kd_loss = renormalized_kd_loss(student_logit, output_teacher_batch)
        kd_loss_1_avg.update(renormal_kd_loss.item(), imgs.size(0))
        
        # OPL:
        out_opl = opl(student_feats.to(DEVICE), labels=labels)
        opl_avg.update(out_opl.item(), imgs.size(0))

        # ICV:
        icv_out_loss = icv(student_feats.to(DEVICE), labels=labels)
        kd_loss_3_avg.update(icv_out_loss.item(), imgs.size(0))
        
        # Total loss
        student_total_loss = student_cross_loss * opt.CEHyper + renormal_kd_loss + out_opl + icv_out_loss
        
        # compute gradient and do SGD step
        student_optimizer.zero_grad()
        student_total_loss.backward()
        student_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.print_freq == 0 or i + 1 == len(student_train_loader):
            print("TRAIN [{:5}][{:5}/{:5}] | Time {:16} Data {:16} Accuracy {:18} Loss(Total) {:16} Loss(Renormalized) {:16} Loss(Intra-Class Variation) {:16} Loss(OPL) {:16}".format(
                  str(e), str(i), str(len(student_train_loader)),
                  "{t.val:.3f} ({t.avg:.3f})".format(t=batch_time),
                  "{t.val:.3f} ({t.avg:.3f})".format(t=data_time),
                  "{t.val:.3f} ({t.avg:.3f})".format(t=acc_avg),
                  "{t.val:.3f} ({t.avg:.3f})".format(t=loss_avg),
                  "{t.val:.3f} ({t.avg:.3f})".format(t=kd_loss_1_avg),
                  "{t.val:.3f} ({t.avg:.3f})".format(t=kd_loss_3_avg),
                  "{t.val:.3f} ({t.avg:.3f})".format(t=opl_avg)))
    #return avg_ce / count, avg_renormalized / count, avg_projected / count, avg_auxiliary / count, avg_opl / count


def validate(data_loader, student_model, criterion, DEVICE, epoch=None):

    batch_time = utils.AverageMeter()
    loss_avg = utils.AverageMeter()
    acc_avg = utils.AverageMeter()
    
    student_model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs, labels = Variable(imgs), Variable(labels)

            # compute output
            logits, _ = student_model(imgs)   # check model
            loss = criterion(logits, labels)
            loss_avg.update(loss.item(), imgs.size(0))
            acc = utils.accuracy(logits, labels)
            acc_avg.update(acc.item(), imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end, imgs.size(0))
            end = time.time()

    '''
    student_model.eval()
    native_class_size = 10
    subclass_factor = 2
    matrix_subclass_distribution = np.zeros((native_class_size, subclass_factor))
    use_subclass_d = True
    val_losses = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            X, y = data[0].to(DEVICE), data[1].to(DEVICE)
            
            logits, _ = student_model(X) # this get's the prediction from the network

            val_losses += criterion(logits, y)

            if use_subclass_d == False:
                predicted_classes = torch.max(logits, 1)[1]  # get class from network's prediction

            else:
                prediction_holder = torch.zeros((logits.shape[0], native_class_size))
                soft_out = torch.nn.functional.softmax(logits, dim=1)
                for bx in range(logits.shape[0]):
                    for ix in range(native_class_size):
                        prediction_holder[bx,ix]=torch.sum(soft_out[bx,ix*(subclass_factor):ix*(subclass_factor)+subclass_factor])
                predicted_classes = torch.max(prediction_holder,1)[1]

            # calculate P/R/F1/A metrics for batch
            #for acc, metric in zip((accuracy),
            #                       (accuracy_score)):

            #for acc, metric in zip((precision, recall, f1, accuracy),
            #                        (precision_score, recall_score, f1_score, accuracy_score)):

            acc=accuracy
            metric=accuracy_score
            #print(metric)
            acc.append(calculate_metric(metric, y.cpu(), predicted_classes.cpu()))

    #val_acc_hist[epoch]=sum(accuracy) / len(accuracy)

    return 0, sum(accuracy) / len(accuracy), 0
    '''
    
    return loss_avg.avg, acc_avg.avg, batch_time


def main():
    
    # get arguments
    opt = get_arguments()
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mode = 'train' if opt.mode == 'train' else 'eval'
    logger = utils.Logger(opt.log2file, mode=mode, model_dir=opt.model_dir, teacher_student='student')

    trueLabel2fakeLabel, fakeLabel2trueLabel = utils.get_tables(opt.studentTask.split('-')[-1],
        opt.teacherClass, opt.studentClass)

    # load datasets (student) 
    student_train_loader = dataloaders.get_dataloader(dataset=opt.studentTask,
            batch_size=opt.batch_size,
            shuffle=True,
            mode='train',
            num_workers=opt.workers, class_num=len(trueLabel2fakeLabel), labelTable=trueLabel2fakeLabel)
    
    student_test_loader = dataloaders.get_dataloader(dataset=opt.studentTask,
            batch_size=opt.batch_size,
            shuffle=False,
            mode='test',
            num_workers=opt.workers, class_num=len(trueLabel2fakeLabel), labelTable=trueLabel2fakeLabel)
    num_classes_student = student_test_loader.dataset.num_classes

    # different modes
    if mode == 'train':
        # load models
        teacher_model = utils.get_model(opt.teacherArch, opt.teacherClass, pretrained=opt.pretrainedStudent)
        teacher_fn = os.path.join(opt.model_dir, opt.teacher_model_name + '.pth.tar')
        teacher_model.load_state_dict(torch.load(teacher_fn)['state_dict'])

        student_model = utils.get_model(opt.studentArch, num_classes_student, pretrained=opt.pretrainedStudent)
        
        teacher_model = teacher_model.to(DEVICE)
        student_model = student_model.to(DEVICE)

    elif mode == 'eval':
        # load model
        student_model = utils.get_model(opt.studentArch, num_classes_student, pretrained=opt.pretrainedStudent)
        student_model = student_model.to(DEVICE)

    # Losses
    student_criterion = nn.CrossEntropyLoss()

    # model training
    if mode == 'train':
        student_optimizer = torch.optim.SGD(student_model.parameters(), lr=opt.lr,
                                            momentum=opt.momentum, weight_decay=opt.weight_decay)

        #interest_classes = torch.tensor([x for x in range(num_classes_student)]).to(DEVICE)
        interest_classes = torch.as_tensor(list(fakeLabel2trueLabel.items()))[:, 1].to(DEVICE)
        
        # setup Renormalized KD loss
        renormalized_kd_loss = RenormalizedKDLoss(device=DEVICE, 
                                                  interest_class=interest_classes, 
                                                  alpha=opt.renormHyper).to(DEVICE)
        
        # setup OPL (Orthogonal Projection Loss)
        opl = OrthogonalProjectionLoss(opl_hyper=opt.oplHyper, device=DEVICE).to(DEVICE)

        # setup Intra-class variation loss
        icv = var_min_ce(icv_hyper=opt.icvHyper, num_classes=opt.studentClass, device=DEVICE).to(DEVICE)
        
        max_test_acc = 0
        for e in range(1, opt.epochs + 1):
            logger.add_line("="*30+"   Train (Epoch {})   ".format(e)+"="*30)
            student_optimizer = utils.adjust_learning_rate(student_optimizer, e, opt.lr, opt.lr_decay_epochs, logger)
            
            # train function
            train(student_train_loader, teacher_model, student_model,
                  student_criterion, student_optimizer, e, DEVICE, opt, renormalized_kd_loss, opl, icv)
            
            # Evaluate on test set
            err, acc, run_time = validate(student_test_loader, student_model, student_criterion, DEVICE, epoch=e)
            
            if acc > max_test_acc:
                max_test_acc = acc
                # save best err and save checkpoint
                utils.save_checkpoint(opt.model_dir,
                                      {'epoch': e,
                                       'state_dict': student_model.state_dict(),
                                       'err': err,
                                       'acc': acc},
                                      teacher_or_student=opt.student_model_name)
            print(f'Max Test Acc: {max_test_acc}')
    # model inference
    elif mode == 'eval':
        fn = os.path.join(opt.model_dir, opt.student_model_name + '.pth.tar')
        student_model.load_state_dict(torch.load(fn)['state_dict'])
        err, acc, run_time = validate(student_test_loader, student_model, student_criterion, DEVICE)

        print('[RUN TIME] {time.avg:.3f} sec/sample'.format(time=run_time))
        print('[FINAL] {name:<30} {loss:.7f}'.format(name='crossentropy', loss=err))
        print('[FINAL] {name:<30} {acc:.7f}'.format(name='accuracy', acc=acc))


if __name__ == '__main__':
    main()
