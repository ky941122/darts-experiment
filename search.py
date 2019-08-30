""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from visualize import plot


config = SearchConfig()

device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

TRAIN_DATA_PATH = "/share/kangyu/speaker/data_for_pytorch/train"
DEV_DAHAI_DATA_PATH = "/share/kangyu/speaker/data_for_pytorch/dev"
TEST_ZHIKANG_DATA_PATH = "/share/kangyu/speaker/data_for_pytorch/test"

def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    # get data with meta info
    dahai_train_dataset = utils.MyDataset(data_dir=TRAIN_DATA_PATH,
                                            )
    dahai_dev_dataset = utils.MyDataset(data_dir=DEV_DAHAI_DATA_PATH,
                                            )
    # zhikang_test_dataset = utils.MyDataset(window_size=WINDOW_SIZE,
    #                                     window_step=WINDOW_STEP_DEV,
    #                                     data_path=TEST_ZHIKANG_DATA_PATH,
    #                                     voice_embed_path=TEST_ZHIKANG_VOICE_EMBEDDING_PATH,
    #                                     w2i=w2i,
    #                                     sent_max_len=SENT_MAX_LEN,
    #                                     )

    train_data = utils.DataProvider(batch_size=config.batch_size, dataset=dahai_train_dataset, is_cuda=config.is_cuda)
    dev_data = utils.DataProvider(batch_size=config.batch_size, dataset=dahai_dev_dataset, is_cuda=config.is_cuda)
    # test_data = utils.DataProvider(batch_size=config.batch_size, dataset=zhikang_test_dataset, is_cuda=config.is_cuda)

    print("train data nums:", len(train_data.dataset), "dev data nums:", len(dev_data.dataset))

    net_crit = nn.CrossEntropyLoss(reduction="none").to(device)
    model = SearchCNNController(config.embedding_dim, config.init_channels, config.n_classes, config.layers,
                                net_crit, config=config, n_nodes=config.n_nodes, device_ids=config.gpus)
    model = model.to(device).float()

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)

    ######  余弦退火-调整学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_acc = 0.
    best_genotype = model.genotype()
    while True:
        epoch = train_data.epoch
        if epoch > config.epochs-1:
            break
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        train(train_data, dev_data, epoch, model, architect, w_optim, alpha_optim, lr)

        # validation
        cur_step = train_data.iteration
        valid_acc = validate(dev_data, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_acc < valid_acc:
            best_acc = valid_acc
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_acc))
    logger.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, epoch, model, architect, w_optim, alpha_optim, lr):
    acc = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = train_loader.iteration
    writer.add_scalar('train/lr', lr, cur_step)

    # 设置成training mode.
    model.train()

    cur_epoch = train_loader.epoch
    assert cur_epoch == epoch
    step = 0
    while cur_epoch == epoch:
        trn_sent_idx, trn_voice_embed, trn_word_num_per_sent, trn_sent_num_per_example, trn_label = train_loader.next()
        trn_sent_idx, trn_voice_embed, trn_word_num_per_sent, trn_sent_num_per_example, trn_label = \
            trn_sent_idx.to(device, non_blocking=True), trn_voice_embed.to(device, non_blocking=True), \
            trn_word_num_per_sent.to(device, non_blocking=True), trn_sent_num_per_example.to(device, non_blocking=True),\
            trn_label.to(device, non_blocking=True)

        val_sent_idx, val_voice_embed, val_word_num_per_sent, val_sent_num_per_example, val_label = valid_loader.next()
        val_sent_idx, val_voice_embed, val_word_num_per_sent, val_sent_num_per_example, val_label = \
            val_sent_idx.to(device, non_blocking=True), val_voice_embed.to(device, non_blocking=True), \
            val_word_num_per_sent.to(device, non_blocking=True), val_sent_num_per_example.to(device, non_blocking=True), \
            val_label.to(device, non_blocking=True)

        trn_sent_num_per_example = trn_sent_num_per_example.squeeze()
        val_sent_num_per_example = val_sent_num_per_example.squeeze()

        cur_epoch = train_loader.epoch

        N = trn_sent_idx.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_sent_idx, trn_label,
                                    val_sent_idx, val_label,
                                    lr, w_optim,
                                    trn_sent_num_per_example, trn_voice_embed,
                                    val_sent_num_per_example, val_voice_embed)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()

        loss, accuracy = model.loss(trn_sent_idx, trn_label, trn_sent_num_per_example, trn_voice_embed)

        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        prec1 = accuracy
        losses.update(loss.item(), N)
        acc.update(prec1.item(), N)

        if step % config.print_freq == 0 or step == len(train_loader.dataset) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Acc: ({acc.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader.dataset)-1, losses=losses,
                    acc=acc))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/acc', prec1.item(), cur_step)
        cur_step += 1
        step += 1

    logger.info("Train: [{:2d}/{}] Final Accuracy {:.4%}".format(epoch+1, config.epochs, acc.avg))


def validate(valid_loader, model, epoch, cur_step):
    acc = utils.AverageMeter()
    losses = utils.AverageMeter()

    init_valid_epoch = valid_loader.epoch
    cur_valid_epoch = valid_loader.epoch
    step = 0

    # This is equivalent with self.train(False).
    model.eval()

    with torch.no_grad():
        while init_valid_epoch == cur_valid_epoch:
            val_sent_idx, val_voice_embed, val_word_num_per_sent, val_sent_num_per_example, val_label = valid_loader.next()
            val_sent_idx, val_voice_embed, val_word_num_per_sent, val_sent_num_per_example, val_label = \
                val_sent_idx.to(device, non_blocking=True), val_voice_embed.to(device, non_blocking=True), \
                val_word_num_per_sent.to(device, non_blocking=True), val_sent_num_per_example.to(device,
                                                                                                 non_blocking=True), \
                val_label.to(device, non_blocking=True)

            val_sent_num_per_example = val_sent_num_per_example.squeeze()

            cur_valid_epoch = valid_loader.epoch

            N = val_sent_idx.size(0)  # 就是返回shape

            loss, accuracy = model.loss(val_sent_idx, val_label, val_sent_num_per_example,
                                        val_voice_embed)

            prec1 = accuracy
            losses.update(loss.item(), N)
            acc.update(prec1.item(), N)

            if step % config.print_freq == 0 or step == len(valid_loader.dataset)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Acc: ({acc.avg:.1%})".format(
                        epoch+1, config.epochs, step, len(valid_loader.dataset)-1, losses=losses,
                        acc=acc))

            step += 1

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/acc', acc.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final Accuracy {:.4%}".format(epoch+1, config.epochs, acc.avg))

    return acc.avg


if __name__ == "__main__":
    main()
