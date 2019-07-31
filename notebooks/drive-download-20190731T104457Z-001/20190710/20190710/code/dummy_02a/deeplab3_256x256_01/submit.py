from train import null_augment
from model import *
from dataset import *
from common import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97225#latest-570515
# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview/evaluation

# ???
def kaggle_metric_one(predict, num_predict, component, num_component):

    if num_component == 0:
        if num_predict == 0:
            return 1, None, None, None, None, None
        else:
            return 0, None, None, None, None, None

    # ----
    if num_predict == 0:
        return 0, None, None, None, None, None

    predict = predict.reshape(1, num_predict, -1)
    truth = component.reshape(num_component, 1, -1)

    intersect = predict*truth
    union = predict+truth

    intersect = intersect.sum(-1)
    union = union.sum(-1)
    dice = 2.0*intersect/(union+1e-12)

    # For each ground truth mask, we take the Dice score of the closest predicted mask, without replacement
    match = np.zeros((num_component, num_predict), np.int32)
    for i in range(num_component):
        sort = np.argsort(-dice[i])
        for j in sort:
            if match[:, j].sum() == 0:
                match[i, j] = 1
                break

    num_matched = match.sum()
    num_unmatched_component = num_component-num_matched
    num_unmatched_predict = num_predict-num_matched

    # Unmatched ground truth and prediction masks are counted as having a Dice score of zero in the average.
    score = (match*dice).sum()/(num_matched +
                                num_unmatched_component+num_unmatched_predict)

    return score, match, dice, num_matched, num_unmatched_component, num_unmatched_predict

#############################################################################################


def npy_to_encoded(test_probability, test_id, csv_file, threshold=0.90, min_size=2400):

    num_test = len(test_id)

    image_id = []
    encoded_pixel = []
    for b in range(num_test):
        id = test_id[b]

        probability = test_probability[b, 0]
        predict, num_predict = mask_to_component(probability, threshold)
        predict = np.array([
            (cv2.resize(p, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.float32) for p in predict
        ])

        # filter off small size -----
        filter = []
        for p in predict:
            if p.sum() > min_size:
                filter.append(p)

        num_predict = len(filter)
        if num_predict > 0:
            predict = np.array(filter)
        else:
            num_predict = 0
            predict = np.zeros((1024, 1024), np.float32)

        # ---

        if num_predict == 0:
            image_id.append(id)
            encoded_pixel.append('-1')
        else:
            for c in predict:
                r = run_length_encode(c)
                image_id.append(id)
                encoded_pixel.append(r)

    return image_id, encoded_pixel


def run_submit():

    out_dir = \
        '/root/share/project/kaggle/2019/chest/result/deeplab3plus-01-256x256'

    initial_checkpoint = \
        '/root/share/project/kaggle/2019/chest/result/deeplab3plus-01-256x256/checkpoint/00210000_model.pth'

    # parameter
    threshold = 0.90
    min_size = 2400

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

    npy_file = out_dir + '/submit/%s/test-probability.npy' % mode
    id_file = out_dir + '/submit/%s/test-id.txt' % mode
    csv_file = out_dir + \
        '/submit/%s/submission-%s.csv' % (mode,
                                          initial_checkpoint.split('/')[-1][:-4])

    # dataset ----------------------------------------
    log.write('** dataset setting **\n')
    batch_size = 4  # *2 #280*2 #256*4 #128 #256 #512  #16 #32

    if mode == 'train':
        test_dataset = XrayDataset(
            mode='train',
            csv='train-rle.csv',
            folder='dicom/dicom-images-train',
            split='valid_600.npy',
            augment=null_augment,
        )
    if mode == 'test':
        test_dataset = XrayDataset(
            mode='test',
            csv='test-rle.csv',
            folder='dicom/dicom-images-test',
            split=None,  # 'test_leak_78.npy',#None,
            augment=null_augment,
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
    test_id = []
    test_probability = []
    test_truth = []
    test_loss = 0
    test_num = 0

    if 1:
        start = timer()
        for b, (input, truth, num_component, infor) in enumerate(test_loader):

            net.eval()
            input = input.cuda()
            truth = truth.cuda()

            with torch.no_grad():
                logit = net(input)
                probability = torch.sigmoid(logit)
                loss = criterion(logit, truth)

            # ---
            batch_size = len(infor)
            test_id.extend([i.image_id for i in infor])
            test_probability.append(probability.data.cpu().numpy())
            test_truth.append(truth.data.cpu().numpy())

            test_loss += batch_size*loss.item()
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
        np.save(npy_file, test_probability)
        write_list_to_file(id_file, test_id)

        if mode == 'train':
            test_loss = test_loss/test_num

            # compute dice loss for whole mask
            t = test_truth > 0.5
            p = test_probability > 0.5
            t = t.reshape(test_num, -1)
            p = p.reshape(test_num, -1)

            t_sum = t.sum(-1)
            p_sum = p.sum(-1)

            pos_index = np.where(t_sum > 0)[0]
            neg_index = np.where(t_sum == 0)[0]

            dice_neg = (p_sum == 0).astype(np.float32)
            dice_pos = 2*(p*t).sum(-1)/((p+t).sum(-1))
            dice_neg = dice_neg[neg_index]
            dice_pos = dice_pos[pos_index]
            dice = np.concatenate([dice_pos, dice_neg])

            dice_neg = dice_neg.mean()
            dice_pos = dice_pos.mean()
            dice = dice.mean().item()
            num_neg = len(neg_index)
            num_pos = len(pos_index)

            log.write('\n')
            log.write('loss = %0.5f\n' % test_loss)
            log.write('dice      = %0.5f\n' % dice)
            log.write('dice_pos  = %0.5f\n' % dice_pos)
            log.write('dice_neg  = %0.5f\n' % dice_neg)
            log.write('num_pos   = %d\n' % num_pos)
            log.write('num_neg   = %d\n' % num_neg)
            log.write('\n')

    # ----
    test_probability = np.load(npy_file)
    test_id = read_list_from_file(id_file)
    num_test = len(test_id)

    image_id, encoded_pixel = npy_to_encoded(
        test_probability, test_id, csv_file, threshold, min_size)

    df = pd.DataFrame(list(zip(image_id, encoded_pixel)),
                      columns=['ImageId', 'EncodedPixels'])
    df.to_csv(csv_file, columns=['ImageId', 'EncodedPixels'], index=False)

    log.write('\n')
    log.write('threshold = %0.5f\n' % (threshold))
    log.write('min_size  = %d\n' % (min_size))
    log.write('\n')
    log.write('id_file   = %s\n' % (id_file))
    log.write('npy_file  = %s\n' % (npy_file))
    log.write('csv_file  = %s\n' % (csv_file))
    log.write('\n')
    log.write('test_id = %d\n' % (len(test_id)))
    log.write('test_probability = %s\n' % (str(test_probability.shape)))
    log.write('\n')


####################################################################################

# <todo> clean up code for finding the best threshold and min mask size
# def run_analysis():
#
#
#     out_dir = \
#         '/root/share/project/kaggle/2019/chest/result/deeplab3plus-01-256x256'
#
#     initial_checkpoint = \
#         '/root/share/project/kaggle/2019/chest/result/deeplab3plus-01-256x256/checkpoint/00210000_model.pth'
#
#     mode = 'train' #debug
#
#
#     ## dataset ----------------------------------------
#     if mode == 'train':
#         test_dataset = XrayDataset(
#             mode   = 'train',
#             csv    = 'train-rle.csv',
#             folder = 'dicom/dicom-images-train',
#             split  = 'valid_600.npy',
#             augment= None,
#         )
#
#     npy_file = out_dir +'/submit/test-probability.npy'
#     id_file  = out_dir +'/submit/test-id.txt'
#     test_probability = np.load(npy_file)
#     test_id = read_list_from_file(id_file)
#
#
#     kaggle_pos = []
#     kaggle_neg = []
#     for b in range(len(test_dataset)):
#         image, component, num_component, infor = test_dataset[b]
#
#         assert(infor.image_id == test_id[b])
#
#
#         probability = test_probability[b,0]
#         predict, num_predict = mask_to_component(probability, threshold=0.90)
#         predict = np.array([
#             (cv2.resize(p, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)>0.5).astype(np.float32) for p in predict
#         ])
#
#         #filter off small size
#         filter = []
#         for p in predict:
#             if p.sum()>2400:
#                 filter.append(p)
#
#         num_predict = len(filter)
#         if num_predict>0:
#             predict = np.array(filter)
#         else:
#             num_predict = 0
#             predict = np.zeros((1024,1024), np.float32)
#
#         print(b, test_id[b], predict.shape, probability.mean(), probability.max())
#         #---
#         score, match, dice, num_matched, num_unmatched_component, num_unmatched_predict = \
#             kaggle_metric_one(predict, num_predict, component, num_component)
#
#         if num_component==0:
#             kaggle_neg.append(score)
#         else:
#             kaggle_pos.append(score)
#         zz=0
#
#     kaggle_neg = np.array(kaggle_neg)
#     kaggle_pos = np.array(kaggle_pos)
#
#     kaggle_neg_score = kaggle_neg.mean()
#     kaggle_pos_score = kaggle_pos.mean()
#
#     print(kaggle_neg)
#     print(kaggle_pos)
#     print(kaggle_neg_score)
#     print(kaggle_pos_score)
#
#     print(0.7886*kaggle_neg_score + (1-0.7886)*kaggle_pos_score)
# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_submit()
    # run_analysis()
