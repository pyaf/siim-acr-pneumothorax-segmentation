from model import *
from dataset import *
from common import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


#
# def load_pretrain(net, pretrain_file):
#
#     #raise NotImplementedError
#     print('\tload pretrain_file: %s'%pretrain_file)
#     pretrain_state_dict = torch.load(pretrain_file)
#     state_dict = net.state_dict()
#
#     keys = list(state_dict.keys())
#
#     i = 0
#     for key in keys:
#         if any(s in key for s in ['logit.',]):
#             continue
#         i=i+1
#         state_dict[key] = pretrain_state_dict[key]
#
#     net.load_state_dict(state_dict)
#     print('')
#     print('len(pretrain_state_dict.keys()) = %d'%len(pretrain_state_dict.keys()))
#     print('len(state_dict.keys())          = %d'%len(state_dict.keys()))
#     print('loaded    = %d'%i)
#     print('')


def valid_augment(image, mask, infor):
    mask = cv2.resize(mask, dsize=None, fx=1 /
                      LOGIT_TO_IMAGE_SCALE, fy=1/LOGIT_TO_IMAGE_SCALE)
    return image, mask, infor


def train_augment(image, mask, infor):
    image, mask = do_random_scale_rotate(image, mask)
    image = do_random_log_contast(image)
    mask = cv2.resize(mask, dsize=None, fx=1 /
                      LOGIT_TO_IMAGE_SCALE, fy=1/LOGIT_TO_IMAGE_SCALE)
    return image, mask, infor


# ------------------------------------

def do_valid(net, valid_loader):

    valid_num = 0
    valid_num_neg = 0
    valid_num_pos = 0

    valid_probability = []
    valid_truth = []

    valid_loss = 0
    valid_dice = 0
    valid_dice_neg = 0
    valid_dice_pos = 0
    for b, (input, truth, infor) in enumerate(valid_loader):

        # if b==5: break
        net.eval()
        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit = net(input)  # data_parallel(net,input)  #net(input)
            probability = torch.sigmoid(logit)
            loss = criterion(logit, truth)
            dice, dice_neg, dice_pos, num_neg, num_pos = metric(logit, truth)

        # ---
        batch_size = len(infor)
        valid_probability.append(probability.data.cpu().numpy())
        valid_truth.append(truth.data.cpu().numpy())

        valid_loss += batch_size*loss.item()
        valid_dice += batch_size*dice
        if num_neg > 0:
            valid_dice_neg += num_neg*dice_neg
        if num_pos > 0:
            valid_dice_pos += num_pos*dice_pos
        valid_num += batch_size
        valid_num_neg += num_neg
        valid_num_pos += num_pos

        print('\r %8d /%8d' %
              (valid_num, len(valid_loader.dataset)), end='', flush=True)

        pass  # -- end of one data loader --
    assert(valid_num == len(valid_loader.dataset))
    # print('')
    valid_loss = valid_loss/valid_num
    valid_dice = valid_dice/valid_num
    valid_dice_neg = valid_dice_neg/valid_num_neg
    valid_dice_pos = valid_dice_pos/valid_num_pos

    return [valid_loss, valid_dice, valid_dice_neg, valid_dice_pos]


def run_train():

    out_dir = \
        '/root/share/project/kaggle/2019/chest/result/unet-256-fold3-00'

    initial_checkpoint = \
        None  # '/root/share/project/kaggle/2019/chest/result/unet-256-fold3-00/checkpoint/00270000_model.pth'
    # '/root/share/project/kaggle/2019/chest/result/hyper-00/checkpoint/00130000_model.pth'
    #

    schduler = NullScheduler(lr=0.01)

    # setup  -----------------------------------------------------------------------------
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/train',  exist_ok=True)
    os.makedirs(out_dir + '/backup', exist_ok=True)
    backup_project_as_zip(PROJECT_PATH, out_dir +
                          '/backup/code.train.%s.zip' % IDENTIFIER)

    log = Logger()
    log.open(out_dir+'/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    # dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 7  # 8
    iter_accum = 4

    train_dataset = XrayDataset(
        mode='train',
        csv=['train-rle.single-mask.csv'],
        split='train3_10075.npy',
        augment=train_augment,
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(train_dataset, 3*len(train_dataset)),
        #sampler    = SequentialSampler(train_dataset),
        #sampler    = RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=null_collate
    )

    valid_dataset = XrayDataset(
        mode='train',
        csv=['train-rle.single-mask.csv'],
        split='valid3_600.npy',
        augment=valid_augment,
    )
    valid_loader = DataLoader(
        valid_dataset,
        #sampler    = SequentialSampler(valid_dataset),
        sampler=RandomSampler(valid_dataset),
        batch_size=2,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=null_collate
    )

    assert(len(train_dataset) >= batch_size)
    log.write('batch_size = %d\n' % (batch_size))
    log.write('train_dataset : \n%s\n' % (train_dataset))
    log.write('valid_dataset : \n%s\n' % (valid_dataset))
    log.write('\n')

    # net ----------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    if initial_checkpoint is not None:
        net.load_state_dict(torch.load(
            initial_checkpoint, map_location=lambda storage, loc: storage), strict=False)

    log.write('%s\n' % (type(net)))
    log.write('\n')

    # optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    # net.set_mode('train',is_freeze_bn=True)
    # -----------------------------------------------

    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    #optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters(
    )), lr=schduler(0), momentum=0.9, weight_decay=0.0001)

    num_iters = 3000*1000
    iter_smooth = 50
    iter_log = 500
    iter_valid = 2500
    iter_save = [0, num_iters-1]\
        + list(range(0, num_iters, 5000))  # 1*1000

    start_iter = 0
    start_epoch = 0
    rate = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace(
            '_model.pth', '_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint = torch.load(initial_optimizer)
            start_iter = checkpoint['iter']
            start_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
        pass

    log.write('optimizer\n  %s\n' % (optimizer))
    log.write('schduler\n  %s\n' % (schduler))
    log.write('\n')

    ## start training here! ##############################################

    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n' % (batch_size, iter_accum))
    log.write('                      |----------- VALID -----------|-------- TRAIN/BATCH -------------------------\n')
    log.write('rate     iter   epoch |  loss    dice   neg   pos   |  loss    dice   neg   pos   |  time          \n')
    log.write('---------------------------------------------------------------------------------------------------\n')
    # 0.00000    0.0*   0.0 |  0.693   0.019  0.00  0.00  |  0.000   0.000  0.00  0.00  |  0 hr 00 min

    train_loss = np.zeros(20, np.float32)
    valid_loss = np.zeros(20, np.float32)
    batch_loss = np.zeros(20, np.float32)
    iter = 0
    i = 0

    start = timer()
    while iter < num_iters:
        sum_train_loss = np.zeros(20, np.float32)
        sum = np.zeros(20, np.float32) + 1e-8

        optimizer.zero_grad()
        for input, truth, infor in train_loader:

            # while 1:
            batch_size = len(infor)
            iter = i + start_iter
            epoch = (iter-start_iter)*batch_size / \
                len(train_dataset) + start_epoch

            # debug-----------------------------
            # if 0:
            #     pass

            # if 0:
            if (iter % iter_valid == 0):
                valid_loss = do_valid(net, valid_loader)
                # pass

            if (iter % iter_log == 0):
                print('\r', end='', flush=True)
                asterisk = '*' if iter in iter_save else ' '
                log.write('%0.5f  %5.1f%s %5.1f |  %5.3f   %5.3f  %4.2f  %4.2f  |  %5.3f   %5.3f  %4.2f  %4.2f  | %s' % (
                    rate, iter/1000, asterisk, epoch,
                    *valid_loss[:4],
                    *train_loss[:4],
                    time_to_str((timer() - start), 'min'))
                )
                log.write('\n')

            # if 0:
            if iter in iter_save:
                torch.save(net.state_dict(), out_dir +
                           '/checkpoint/%08d_model.pth' % (iter))
                torch.save({
                    # 'optimizer': optimizer.state_dict(),
                    'iter': iter,
                    'epoch': epoch,
                }, out_dir + '/checkpoint/%08d_optimizer.pth' % (iter))
                pass

            # learning rate schduler -------------
            lr = schduler(iter)
            if lr < 0:
                break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            # one iteration update  -------------
            # net.set_mode('train',is_freeze_bn=True)

            net.train()
            input = input.cuda()
            truth = truth.cuda()

            logit = net(input)  # data_parallel(net,input)  #net(input)
            loss = criterion(logit, truth)
            dice, dice_neg, dice_pos, num_neg, num_pos = metric(logit, truth)

            (loss/iter_accum).backward()
            if (iter % iter_accum) == 0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  ------------
            batch_loss[:4] = [loss.item(), dice, dice_neg, dice_pos]
            sum_train_loss[:4] += [loss.item()*batch_size, dice *
                                   batch_size, dice_neg*num_neg, dice_pos*num_pos]
            sum[:4] += [batch_size, batch_size, num_neg, num_pos]
            if iter % iter_smooth == 0:
                train_loss = sum_train_loss/sum
                sum_train_loss[...] = 0
                sum[...] = 1e-8

            print('\r', end='', flush=True)
            asterisk = ' '
            print('%0.5f  %5.1f%s %5.1f |  %5.3f   %5.3f  %4.2f  %4.2f  |  %5.3f   %5.3f  %4.2f  %4.2f  | %s' % (
                rate, iter/1000, asterisk, epoch,
                *valid_loss[:4],
                *batch_loss[:4],
                time_to_str((timer() - start), 'min')), end='', flush=True)
            i = i+1

            # debug-----------------------------
            # if 0:
            for di in range(10):
                if (iter+di) % 5000 == 0:
                    pass
                    probability = torch.sigmoid(logit)
                    # probability = F.interpolate(probability, size=(1024,1024))
                    # truth       = F.interpolate(truth, size=(1024,1024))

                    input = input.data.cpu().numpy()*255
                    input = input.astype(np.uint8)
                    truth = truth.data.cpu().numpy()
                    probability = probability.data.cpu().numpy()

                    for b in range(batch_size):
                        overlay = draw_result_overlay(
                            input[b, 0], truth[b, 0], probability[b, 0])

                        image_show('overlay', overlay, 1)
                        cv2.imwrite(out_dir + '/train/%05d.png' %
                                    (di*100+b), overlay)
                        cv2.waitKey(1)
                        pass

        pass  # -- end of one data loader --
    pass  # -- end of all iterations --

    log.write('\n')


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_train()
