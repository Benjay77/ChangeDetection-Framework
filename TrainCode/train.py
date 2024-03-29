"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 训练模块
"""
import argparse
import os
import time
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import prepare_data, Dataset
from utils.my_net import MyNet
from utils.plot import argparses_plot
from val import val


def getargs():
    parse = argparse.ArgumentParser(description = "Change_Detection_Train")
    parse.add_argument('--arch', '-a', metavar = 'ARCH', default = 'SE_Resnet50',
                       help = 'SE_Resnet50/SK_Resnet50')
    parse.add_argument("--dataset_dir", type = str, default = './dataset')
    parse.add_argument("--preprocess", type = bool, default = False
                       , help = 'run prepare_data or not')
    parse.add_argument('--resolution', type = str, default = '')
    parse.add_argument("--interpret_type", type = str, default = '')
    parse.add_argument("--weight_name", type = str, default = time.strftime("%Y%m%d%H%M", time.localtime()))
    parse.add_argument("--image_size", type = int, default = 512, help = 'size of image')
    parse.add_argument('--image_type', type = str, default = '.tif')
    parse.add_argument('--mask_type', type = str, default = '.png')
    parse.add_argument("--batchsize_per_card", type = int, default = 24)
    parse.add_argument('--use_multiple_GPU', type = bool, default = True)
    parse.add_argument("--epochs", type = int, default = 500, help = "Number of training epochs")
    parse.add_argument('--loss_function', type = str, default = 'CACLoss',
                       help = 'CACLoss')
    parse.add_argument("--lr_init", type = float, default = 0.03, help = "Initial learning rate")
    parse.add_argument('--lrf', type = float, default = 0.2)
    parse.add_argument("--kronecker_r1", nargs = '+', type = int,
                       default = [2, 4, 8])
    parse.add_argument("--kronecker_r2", nargs = '+', type = int,
                       default = [1, 3, 5])
    parse.add_argument("--train_epoch_best_loss", type = float, default = 100)
    parse.add_argument("--val_epoch_best_loss", type = float, default = 100)
    parse.add_argument("--log_dir", type = str, default = "./logs", help = 'path of log files')
    parse.add_argument("--tensorboard", type = str, default = "tensorboard", help = 'path of tensorboard files')
    parse.add_argument("--weights", type = str, default = "./inference/weights", help = 'path of weights files')
    parse.add_argument('--resume', type = bool, default = True)
    parse.add_argument('--resume_weight_name', type=str, default='202109151535')
    parse.add_argument('--classes', type = int, default = 2)
    parse.add_argument('--update_lr_epoch', type = int, default = 6)
    parse.add_argument('--early_stop_epoch', type = int, default = 10)
    parse.add_argument('--num_workers', type = int, default = 8)
    parse.add_argument('--device', default = '0', help = 'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parse.parse_args()
    return args


def main():
    # Build model
    model = MyNet(args)

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset('train')
    dataset_val = Dataset('val')
    loader_train = DataLoader(dataset = dataset_train, num_workers = args.num_workers, batch_size = model.batch_size,
                              shuffle = True)
    print("training samples: %d\n" % int(len(dataset_train)))
    print("val samples: %d\n" % int(len(dataset_val)))

    # tensorboardX
    # writer = SummaryWriter(opt.tensorboard)
    # log
    log_path = os.path.join(args.log_dir, args.resolution, args.interpret_type)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    my_log = open(os.path.join(log_path, args.interpret_type + args.weight_name + '.txt'), 'w')

    # training
    if not os.path.exists(os.path.join(args.weights, args.resolution, args.interpret_type,
                                       args.interpret_type + args.weight_name)):
        os.makedirs(os.path.join(args.weights, args.resolution, args.interpret_type,
                                 args.interpret_type + args.weight_name))
    tic = time.time()
    begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    my_log.write(str('training begin time: ') + begin_time + '\n')
    print('training begin time: ', begin_time)
    model_info = str('training model: ') + str(model.net) + ';' + '\n'
    learning_rate_info = str(
        'training lr_init: ') + str(args.lr_init) + ';' + str('training lrf: ') + str(args.lrf) + ';'
    training_info = str(
        'training loss_function: ') + args.loss_function + ';' + str(
        'training dataset: ') + args.dataset_dir.split('/dataset/')[
                        -1] + ';' + str('training image_size: ') + str(
        args.image_size) + ';' + learning_rate_info + str(
        'training batch_size: ') + str(model.batch_size) + ';' + str(
        'training kronecker_r1: ') + str(args.kronecker_r1) + ';' + str(
        'training kronecker_r2: ') + str(args.kronecker_r2) + ';' + model_info + '\n'
    if args.resume:
        training_info = str('resume:') + str(args.resume) + ';' + str(
            'resume_weight_name:') + args.resume_weight_name + ';' + training_info
    my_log.write(training_info)
    print(training_info)

    no_optim = 0
    start_epoch = 0
    end_epoch = 0
    old_lr = new_lr = args.lr_init
    train_loss_list = []
    val_loss_list = []
    model_path = os.path.join(args.weights, args.resolution, args.interpret_type,
                              args.interpret_type + args.weight_name,
                              args.interpret_type + args.weight_name)

    if args.resume:
        resume_model_path = os.path.join(args.weights, args.resolution, args.interpret_type,
                                         args.interpret_type + args.resume_weight_name,
                                         args.interpret_type + args.resume_weight_name)
        resume_model = model.load(resume_model_path + '_last.pth')
        model.net.load_state_dict(resume_model['net'])
        model.optimizer.load_state_dict(resume_model['optimizer'])
        start_epoch = resume_model['cur_epoch']
        train_loss_list = resume_model['train_loss_list']
        val_loss_list = resume_model['val_loss_list']
        if len(train_loss_list) > len(val_loss_list):
            del train_loss_list[-1]

    for epoch in range(start_epoch + 1, args.epochs + 1):

        # train
        train_epoch_loss = 0
        loop = tqdm(enumerate(loader_train), total = len(loader_train))
        for i, (data_prev, data_now, label, _) in loop:
            model.set_input(data_prev, data_now, label)
            train_loss = model.optimize()
            train_epoch_loss += train_loss
            cuda_mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            loop.set_description(f'Epoch[{epoch}/{args.epochs}]  CUDA  {cuda_mem}')
            loop.set_postfix(loss = train_loss)

        # Scheduler and log lr
        old_lr = model.old_lr
        # train_epoch_loss
        train_epoch_loss /= len(loader_train)
        train_loss_list.append(train_epoch_loss)
        # save model
        save_model = {'net': model.net.state_dict(), 'optimizer': model.optimizer.state_dict(),
                      'cur_epoch': epoch, 'train_loss_list': train_loss_list,'val_loss_list': val_loss_list}
        model.save(model_path + '_last.pth', save_model)

        # validate
        loader_val = DataLoader(dataset = dataset_val, num_workers = args.num_workers, batch_size = model.batch_size,
                                shuffle = True)

        val_epoch_loss = val(args, loader_val, model_path + '_last.pth')
        val_loss_list.append(val_epoch_loss)
        save_model['val_loss_list'] = val_loss_list
        model.save(model_path + '_last.pth', save_model)

        # log the images
        # Img = utils.make_grid(img_train[3:,:,:].data, nrow=8, normalize=True, scale_each=True)
        # label_train = torch.unsqueeze(label_train, dim=1)
        # label = utils.make_grid(label_train.float().data, nrow=8, normalize=True, scale_each=True)
        # result = utils.make_grid(result.float().data, nrow=8, normalize=True, scale_each=True)
        # # writer.add_image('image', Img, epoch)
        # writer.add_image('Label', label, epoch)
        # writer.add_image('Result', result, epoch)

        end_epoch = epoch

        if train_epoch_loss < args.train_epoch_best_loss and val_epoch_loss < args.val_epoch_best_loss:
            no_optim = 0
            args.train_epoch_best_loss = train_epoch_loss
            args.val_epoch_best_loss = val_epoch_loss
            model.save(model_path + '_best.pth', None, True)
        else:
            no_optim += 1
            # update learning-rate
            if no_optim > args.update_lr_epoch:
                if old_lr < 5e-7:
                    break
                new_lr = model.update_lr(args.lrf, factor = True, my_log = my_log)

        my_log.write('********************' + '\n')
        update_lr_info = '  --update learning rate: ' + str(old_lr) + ' -> ' + str(
            new_lr)
        log = '--epoch: ' + str(epoch) + '  --time: ' + str(
            int(time.time() - tic)) + update_lr_info + '  --no_optim: ' + str(
            no_optim) + '  --train_epoch_best_loss: ' + str(
            args.train_epoch_best_loss) + '  --val_epoch_best_loss: ' + str(
            args.val_epoch_best_loss) + '  --train_epoch_loss: ' + str(
            train_epoch_loss) + '  --val_epoch_loss: ' + str(
            val_epoch_loss) + '\n'
        my_log.write(log)
        print(log)

        # EarlyStopping
        if no_optim > args.early_stop_epoch:
            my_log.write('early stop at %d epoch' % epoch + '\n')
            print('early stop at %d epoch' % epoch + '\n')
            break
        my_log.flush()

    argparses_plot(args, 'train_loss&val_loss', end_epoch, train_loss_list, val_loss_list)
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f'\ntraining end_time: ', end_time)
    my_log.write(str('training end time: ') + end_time + '\n')
    my_log.write(f'{end_epoch} epochs completed in {(time.time() - tic) / 3600:.3f} hours.' + '\n')
    print(f'\n{end_epoch} epochs completed in {(time.time() - tic) / 3600:.3f} hours.')
    my_log.write('Train Finish' + '\n')
    print(f'\nTrain Finish!')
    torch.cuda.empty_cache()
    my_log.close()
    # writer.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = getargs()
    if args.preprocess:
        prepare_data(args, 0)
    main()