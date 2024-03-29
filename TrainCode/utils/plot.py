"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/9/3 下午12:50
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/9/3 下午12:50
 *@Description: 训练可视化
"""
import os

import matplotlib.pyplot as plt


def loss_plot(args, loss, fact_epoch):
    num = fact_epoch if args.epochs != fact_epoch else args.epochs
    x = [i for i in range(num)]
    plot_save_path = r'plot/' + str(args.interpret_type + args.weight_name) + '/'
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_loss = plot_save_path + str(args.interpret_type + args.weight_name) + '_loss.jpg'
    plt.figure()
    plt.plot(x, loss, label = 'loss')
    plt.legend()
    plt.savefig(save_loss)


def argparses_plot(args, name, fact_epoch, *argparses):
    num = fact_epoch if args.epochs != fact_epoch else args.epochs
    names = name.split('&')
    argparses_value = argparses
    i = 0
    x = [i for i in range(num)]
    plot_save_path = os.path.join('plot', args.resolution, args.interpret_type, args.interpret_type +
                                  args.weight_name)
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
    save_metrics = os.path.join(plot_save_path, args.interpret_type + args.weight_name + '_' + name + '.jpg')
    plt.figure()
    for y in argparses_value:
        plt.plot(x, y, label = str(names[i]))
        # plt.scatter(x,l,label=str(l))
        i += 1
    plt.legend()
    plt.savefig(save_metrics)