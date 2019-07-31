from model import *
from dataset import *
from common import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


#from train import valid_augment


def valid_augment(image, mask, infor):
    mask = cv2.resize(mask, dsize=None, fx=1 /
                      LOGIT_TO_IMAGE_SCALE, fy=1/LOGIT_TO_IMAGE_SCALE)
    return image, mask, infor


# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97225#latest-570515
# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview/evaluation

#############################################################################################
def post_process(probability, threshold, min_size):

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predict = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predict[p] = 1
            num += 1
    return predict, num


def compute_metric(test_id, test_truth, test_probability):

    test_num = len(test_truth)
    truth = test_truth.reshape(test_num, -1)
    probability = test_probability.reshape(test_num, -1)

    loss = - truth*np.log(probability) - (1-truth)*np.log(1-probability)
    loss = loss.mean()

    t = (truth > 0.5).astype(np.float32)
    p = (probability > 0.5).astype(np.float32)
    t_sum = t.sum(-1)
    p_sum = p.sum(-1)
    neg_index = np.where(t_sum == 0)[0]
    pos_index = np.where(t_sum >= 1)[0]

    dice_neg = (p_sum == 0).astype(np.float32)
    dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1)+1e-12)
    dice_neg = dice_neg[neg_index]
    dice_pos = dice_pos[pos_index]
    dice = np.concatenate([dice_pos, dice_neg])

    dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
    dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
    dice = dice.mean()

    return loss, dice, dice_neg, dice_pos


def compute_kaggle_lb(test_id, test_truth, test_probability, threshold, min_size):

    test_num = len(test_truth)

    kaggle_pos = []
    kaggle_neg = []
    for b in range(test_num):
        truth = test_truth[b, 0]
        probability = test_probability[b, 0]

        if truth.shape != (1024, 1024):
            truth = cv2.resize(truth, dsize=(1024, 1024),
                               interpolation=cv2.INTER_LINEAR)
            truth = (truth > 0.5).astype(np.float32)

        if probability.shape != (1024, 1024):
            probability = cv2.resize(probability, dsize=(
                1024, 1024), interpolation=cv2.INTER_LINEAR)

        # -----
        predict, num_component = post_process(probability, threshold, min_size)

        score = kaggle_metric_one(predict, truth)
        print('\r%3d  %-56s  %s   %0.5f  %0.5f' %
              (b, test_id[b], predict.shape, probability.mean(), probability.max()), end='', flush=True)

        if truth.sum() == 0:
            kaggle_neg.append(score)
        else:
            kaggle_pos.append(score)

    print('')
    kaggle_neg = np.array(kaggle_neg)
    kaggle_pos = np.array(kaggle_pos)
    kaggle_neg_score = kaggle_neg.mean()
    kaggle_pos_score = kaggle_pos.mean()
    kaggle_score = 0.7886*kaggle_neg_score + (1-0.7886)*kaggle_pos_score

    return kaggle_score, kaggle_neg_score, kaggle_pos_score

###################################################################################


def do_evaluate(net, test_loader):

    test_id = []
    test_probability = []
    test_truth = []
    test_num = 0

    start = timer()
    for b, (input, truth, infor) in enumerate(test_loader):

        net.eval()
        input = input.cuda()
        truth = truth.cuda()

        with torch.no_grad():
            logit = net(input)
            probability = torch.sigmoid(logit)

            # batch_size, C, H, W = probability.shape
            # if H!=1024 or W!=1024:
            #     probability = F.interpolate(probability,size=(1024,1024), mode='bilinear', align_corners=False)

        # ---
        batch_size = len(infor)
        test_id.extend([i.image_id for i in infor])
        test_probability.append(probability.data.cpu().numpy())
        test_truth.append(truth.data.cpu().numpy())
        test_num += batch_size

        # ---
        print('\r %4d / %4d  %s' % (
            test_num, len(test_loader.dataset), time_to_str(
                (timer() - start), 'min')
        ), end='', flush=True)

    assert(test_num == len(test_loader.dataset))
    print('')

    test_truth = np.concatenate(test_truth)
    test_probability = np.concatenate(test_probability)

    return test_probability, test_truth, test_id


def run_submit():

    out_dir = \
        '/root/share/project/kaggle/2019/chest/result/unet-256-fold2-00'

    initial_checkpoint = \
        '/root/share/project/kaggle/2019/chest/result/unet-256-fold2-00/checkpoint/00270000_model.pth'

    # parameter
    threshold = 0.90
    min_size = 3500

    # mode = 'train' #debug
    mode = 'test'

    # setup  -----------------------------------------------------------------------------

    os.makedirs(out_dir + '/submit/%s' % mode, exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    id_file = out_dir + '/submit/%s/test-id.txt' % mode
    predict_file = out_dir + '/submit/%s/test-probability.npy' % mode
    truth_file = out_dir + '/submit/%s/test-truth.npy' % mode
    csv_file = out_dir + '/submit/%s/submission-%s-th%0.2f.csv' % (
        mode, initial_checkpoint.split('/')[-1][:-4], threshold)

    # dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 4  # *2 #280*2 #256*4 #128 #256 #512  #16 #32

    if mode == 'train':
        test_dataset = XrayDataset(
            mode='train',
            csv=['train-rle.single-mask.csv'],
            split='valid2_600.npy',
            augment=valid_augment,
        )
    if mode == 'test':
        test_dataset = XrayDataset(
            mode='test',
            csv=['test-rle.single-mask.csv'],
            split='test_1372.npy',
            augment=valid_augment,
        )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        #sampler     = RandomSampler(test_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=null_collate
    )
    log.write('batch_size = %d\n' % (batch_size))
    log.write('test_dataset : \n%s\n' % (test_dataset))
    log.write('\n')

    # net ----------------------------------------
    log.write('** net setting **\n')
    net = Net().cuda()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    net.load_state_dict(torch.load(initial_checkpoint,
                                   map_location=lambda storage, loc: storage))

    log.write('%s\n' % (type(net)))
    log.write('\n')

    ## start testing here! ##############################################
    if 1:
        test_probability, test_truth, test_id = do_evaluate(net, test_loader)

        write_list_to_file(id_file, test_id)
        np.save(predict_file, test_probability)
        np.save(truth_file, test_truth)

    # ---
    test_id = read_list_from_file(id_file)
    test_probability = np.load(predict_file)
    test_truth = np.load(truth_file)
    num_test = len(test_id)

    if mode == 'train':
        loss, dice, dice_neg, dice_pos = \
            compute_metric(test_id, test_truth, test_probability)

        log.write('\n')
        log.write('loss     = %0.5f\n' % (loss))
        log.write('dice_neg = %0.5f\n' % (dice_neg))
        log.write('dice_pos = %0.5f\n' % (dice_pos))
        log.write('dice     = %0.5f\n' % (dice))
        log.write('\n')

        kaggle_score, kaggle_neg_score, kaggle_pos_score = \
            compute_kaggle_lb(test_id, test_truth,
                              test_probability, threshold, min_size)

        log.write('\n')
        log.write('initial_checkpoint = %s\n' % initial_checkpoint)
        log.write('threshold = %0.6f\n' % threshold)
        log.write('min_size  = %d\n' % min_size)
        log.write('\n')
        log.write('kaggle_neg_score = %0.5f\n' % (kaggle_neg_score))
        log.write('kaggle_pos_score = %0.5f\n' % (kaggle_pos_score))
        log.write('kaggle_score     = %0.5f\n' % (kaggle_score))
        log.write('\n')

    # ===================================================================
    if 0:

        image_id = test_id
        encoded_pixel = []
        for b in range(num_test):
            probability = test_probability[b, 0]
            if probability.shape != (1024, 1024):
                probability = cv2.resize(probability, dsize=(
                    1024, 1024), interpolation=cv2.INTER_LINEAR)

            predict, num_predict = post_process(
                probability, threshold, min_size)

            if num_predict == 0:
                encoded_pixel.append('-1')
            else:
                r = run_length_encode(predict)
                encoded_pixel.append(r)

        df = pd.DataFrame(list(zip(image_id, encoded_pixel)),
                          columns=['ImageId', 'EncodedPixels'])
        df.to_csv(csv_file, columns=['ImageId', 'EncodedPixels'], index=False)

        log.write('\n')
        log.write('threshold = %0.5f\n' % (threshold))
        log.write('min_size  = %d\n' % (min_size))
        log.write('\n')
        log.write('id_file      = %s\n' % (id_file))
        log.write('predict_file = %s\n' % (predict_file))
        log.write('csv_file     = %s\n' % (csv_file))
        log.write('\n')
        log.write('test_id = %d\n' % (len(test_id)))
        log.write('test_probability = %s\n' % (str(test_probability.shape)))
        log.write('\n')


####################################################################################
'''
threshold = 0.900000
min_size  = 3500

kaggle_neg_score  :  0.983333
kaggle_pos_score  :  0.286285
kaggle_score      :  0.835977

'''


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_submit()
