import os
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable

import dataloaders
import utils

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='cifar10', help='task to train (teacher)')
    parser.add_argument('--arch', default='resnet34', help='teacher architecture (default: resnet34)')
    parser.add_argument('--mode', default='train', help='mode to training or evaluation.')
    parser.add_argument('--model_dir', default='./checkpoints', help='model directory')
    parser.add_argument('--log2file', action='store_true', help='log output to file (under model_dir/train.log)')
    parser.add_argument('--pretrained', action='store_true', help='download a pre-trained model')
    parser.add_argument('--classNum', type=int, default=10, help='number of classes')
    
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--lr-decay-epochs', default='60,120,160', help='number of epochs for each lr decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=0.0005, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--print-freq', default=200, type=int, help='print frequency (default: 10 iter)')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    return parser.parse_args()


def train(data_loader, model, criterion, optimizer, epoch, DEVICE, opt):
    
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    loss_avg = utils.AverageMeter()
    acc_avg = utils.AverageMeter()

    model.train()
    
    end = time.time()
    for i, (imgs, labels) in enumerate(data_loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, labels = Variable(imgs), Variable(labels)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        logit, _ = model(imgs)  # check model
        loss = criterion(logit, labels)
        loss_avg.update(loss.item(), imgs.size(0))
        acc = utils.accuracy(logit, labels)
        acc_avg.update(acc.item(), imgs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % opt.print_freq == 0 or i + 1 == len(data_loader):
            print("TRAIN [{:5}][{:5}/{:5}] | Time {:16} Data {:16} Accuracy {:18} Loss {:16}".format(
                  str(epoch), str(i), str(len(data_loader)), 
                  "{t.val:.3f} ({t.avg:.3f})".format(t=batch_time),
                  "{t.val:.3f} ({t.avg:.3f})".format(t=data_time),
                  "{t.val:.3f} ({t.avg:.3f})".format(t=acc_avg),
                  "{t.val:.3f} ({t.avg:.3f})".format(t=loss_avg)))


def validate(data_loader, model, criterion, DEVICE, epoch=None, opt=None, trueLabel2fakeLabel=None):
    
    nb_classes = opt.classNum
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    batch_time = utils.AverageMeter()
    loss_avg = utils.AverageMeter()
    acc_avg = utils.AverageMeter()
    
    model.eval()
    
    count = 0
    correct = 0
    with torch.no_grad():
        end = time.time()
        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs, labels = Variable(imgs), Variable(labels)

            # compute output
            logits, _ = model(imgs)   # check model
            loss = criterion(logits, labels)
            loss_avg.update(loss.item(), imgs.size(0))
            acc = utils.accuracy(logits, labels)
            acc_avg.update(acc.item(), imgs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end, imgs.size(0))
            end = time.time()

            '''
            # compute predictions
            _, preds = torch.max(logits, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                if t.item() in list(trueLabel2fakeLabel.keys()):
                    if t.item() == p.item():
                        correct += 1
                    count += 1
                    confusion_matrix[t.long(), p.long()] += 1
            '''
    '''
    print((correct / count) * 100)
    #print(confusion_matrix.diag())
    print(count)
    print(trueLabel2fakeLabel)
    correct_count = 0
    for class_id in list(trueLabel2fakeLabel.keys()):
        correct_count += confusion_matrix.diag()[class_id]
    print('Subclass Accuracy:', (correct_count / count) * 100)
    '''

    #class_wise_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    # find the worst 5 classes
    #worst_classes = [(idx, class_wise_acc[idx].item()) for idx in range(opt.classNum)]
    #sorted_list = sorted(worst_classes, key=lambda x: x[1])
    #print('Worst Subclasses', sorted_list[:5])

    return loss_avg.avg, acc_avg.avg, batch_time


def main():

    # get arguments
    opt = get_arguments()
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if not os.path.exists(opt.model_dir):
        os.mkdir(opt.model_dir)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mode = 'train' if opt.mode == 'train' else 'eval'
    logger = utils.Logger(opt.log2file, mode=mode, model_dir=opt.model_dir)
     
    # load data
    train_loader = dataloaders.get_dataloader(dataset=opt.task,
            batch_size=opt.batch_size,
            shuffle=True,
            mode='train',
            num_workers=opt.workers, class_num=opt.classNum)

    test_loader = dataloaders.get_dataloader(dataset=opt.task,
            batch_size=opt.batch_size,
            shuffle=False,
            mode='test',
            num_workers=opt.workers, class_num=opt.classNum)
    
    # load model
    model = utils.get_model(opt.arch, pretrained=opt.pretrained, num_classes=opt.classNum)
    model = model.to(DEVICE)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # model training
    save_model_name = 'teacher-' + opt.task + '-' + opt.arch
    if mode == 'train':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr,
                                    momentum=opt.momentum, weight_decay=opt.weight_decay)
        max_test_acc = 0
        for e in range(1, opt.epochs + 1):
            logger.add_line("="*30+"   Train (Epoch {})   ".format(e)+"="*30)
            optimizer = utils.adjust_learning_rate(optimizer, e, opt.lr, opt.lr_decay_epochs, logger)
            # train function
            train(train_loader, model, criterion, optimizer, e, DEVICE, opt)

            # Evaluate on test set
            err, acc, run_time = validate(test_loader, model, criterion, DEVICE, e, opt, None)

            # save the model with the best test acc
            if acc > max_test_acc:
                max_test_acc = acc

                # save best err and save checkpoint
                utils.save_checkpoint(opt.model_dir,
                                      {'epoch': e,
                                       'state_dict': model.state_dict(),
                                       'err': err,
                                       'acc': acc},
                                      teacher_or_student=save_model_name)
            print(f'Max Test Acc: {max_test_acc}')

    elif mode == 'eval':
        fn = opt.model_dir + '/' + save_model_name + '-checkpoint.pth.tar'
        model.load_state_dict(torch.load(fn)['state_dict'])
        err, acc, run_time = validate(test_loader, model, criterion, DEVICE, None, opt)
        
        print('[RUN TIME] {time.avg:.3f} sec/sample'.format(time=run_time))
        print('[FINAL] {name:<30} {loss:.7f}'.format(name='crossentropy', loss=err))
        print('[FINAL] {name:<30} {acc:.7f}'.format(name='accuracy', acc=acc))
        #print('Class-Wise Accuracy', class_wise_acc)


if __name__ == '__main__':
    main()
