from common import *
from kaggle import *

DATA_DIR = '/root/share/project/kaggle/2019/chest/data'


class XrayDataset(Dataset):
    def __init__(self, split, csv, mode, augment=None):

        self.split = split
        self.csv = csv
        self.mode = mode
        self.augment = augment

        self.uid = list(np.load(DATA_DIR + '/split/%s' %
                                split, allow_pickle=True))

        df = [pd.read_csv(DATA_DIR + '/%s' %
                          f, dtype={'EncodedPixels': str}) for f in csv]
        df = pd.concat(df, ignore_index=True)
        self.df = df_loc_by_list(
            df, 'ImageId', [i.split('/')[-1] for i in self.uid])
        self.label = (self.df['EncodedPixels'] != '-1').values.astype(np.int32)

    def __str__(self):
        string = ''
        if 1:
            string += '\tmode    = %s\n' % self.mode
            string += '\tsplit   = %s\n' % self.split
            string += '\tcsv     = %s\n' % str(self.csv)

        if self.mode == 'train':
            string += '\tlabel:1 = %d\n' % (sum(self.label == 1))
            string += '\tlabel:0 = %d\n' % (sum(self.label == 0))
        if self.mode == 'test':
            string += '\tlabel:1 = ?\n'
            string += '\tlabel:0 = ?\n'

        if 1:
            string += '\tlen     = %d\n' % len(self)

        return string

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, index):
        # print(index)

        i = self.uid[index]
        image_id = i.split('/')[-1]

        image = cv2.imread(DATA_DIR + '/png/%s.png' % i, cv2.IMREAD_GRAYSCALE)
        mask = np.zeros((1024, 1024), np.float32)
        label = self.label[index]

        if label == 1:
            rle = self.df.loc[self.df['ImageId'] ==
                              image_id, 'EncodedPixels'].values[0]
            mask = run_length_decode(
                rle, height=1024, width=1024, fill_value=1)

        infor = Struct(
            index=index,
            uid=i,
            image_id=i.split('/')[-1],
            label=label,
            #df = df,
            #is_copy = True,
        )

        if self.augment is None:
            return image, mask, infor
        else:
            return self.augment(image, mask, infor)

##############################################################


# data augmentation
def do_resize(image, mask, fx, fy):
    image = cv2.resize(image, dsize=None, fx=fx, fy=fy)
    mask = cv2.resize(mask, dsize=None, fx=fx, fy=fy)
    return image, mask


def do_center_crop(image, mask, w, h):
    height, width = image.shape
    x = (width - w)//2
    y = (height-h)//2
    image = image[y:y+h, x:x+w]
    mask = mask[y:y+h, x:x+w]
    return image, mask


def do_random_crop(image, mask, w, h):
    height, width = image.shape
    x = np.random.choice(width - w)
    y = np.random.choice(height-h)
    image = image[y:y+h, x:x+w]
    mask = mask[y:y+h, x:x+w]
    return image, mask


def do_flip(image, mask):
    image = cv2.flip(image, 1)
    mask = cv2.flip(mask, 1)
    return image, mask


def do_random_scale_rotate(image, mask):
    dangle = np.random.uniform(-15, 15)
    dscale = np.random.uniform(-0.10, 0.10, 2)
    dshift = np.random.uniform(-0.05, 0.05, 2)

    cos = np.cos(dangle/180*PI)
    sin = np.sin(dangle/180*PI)
    sx, sy = 1 + dscale
    tx, ty = dshift
    cx, cy = 0.5, 0.5

    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    src = src-dshift
    x = ((src-[cx, cy])*[cos, -sin]).sum(1)*sx + tx + cx
    y = ((src-[cx, cy])*[sin, cos]).sum(1)*sy + ty + cy
    src = np.column_stack([x, y])
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    if 1:
        h, w = image.shape
        s = src*[w, h]
        d = dst*[w, h]
        s = s.astype(np.float32)
        d = d.astype(np.float32)
        transform = cv2.getPerspectiveTransform(s, d)
        image = cv2.warpPerspective(image, transform, (w, h),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    if 1:
        h, w = mask.shape
        s = src*[w, h]
        d = dst*[w, h]
        s = s.astype(np.float32)
        d = d.astype(np.float32)
        transform = cv2.getPerspectiveTransform(s, d)
        mask = cv2.warpPerspective(mask, transform, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return image, mask


# https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.adjust_sigmoid
# def do_random_contast(image):
#     cutoff = np.random.uniform(-0.05,0.05,1)
#     gain   = np.random.uniform(0.9,1,1) #5
#
#     image = image.astype(np.float32)/255
#     image = 1/(1 + np.exp(gain*(cutoff - image)))
#    image = np.clip(image*255,0,255).astype(np.uint8)
#     return image


def do_random_gamma_contast(image):
    gamma = np.random.uniform(0.75, 1, 1)
    gain = np.random.uniform(0.9, 1.1, 1)

    image = image.astype(np.float32)/255
    image = gain*image**gamma
    image = np.clip(image*255, 0, 255).astype(np.uint8)
    return image


def do_random_log_contast(image):
    gain = np.random.uniform(0.70, 1.30, 1)
    inverse = np.random.choice(2, 1)

    image = image.astype(np.float32)/255
    if inverse == 0:
        image = gain*np.log(image+1)
    else:
        image = gain*(2**image-1)

    image = np.clip(image*255, 0, 255).astype(np.uint8)
    return image

##############################################################


def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
        infor.append(batch[b][2])

    input = np.stack(input).astype(np.float32)/255
    truth = np.stack(truth).astype(np.float32)
    input = torch.from_numpy(input).unsqueeze(1).float()
    truth = torch.from_numpy(truth).unsqueeze(1).float()

    return input, truth, infor


class BalanceClassSampler(Sampler):

    def __init__(self, dataset, length=None):
        self.dataset = dataset

        if length is None:
            length = len(self.dataset)

        self.length = length

    def __iter__(self):
        pos_index = np.where(self.dataset.label == 1)[0]
        neg_index = np.where(self.dataset.label == 0)[0]
        half = self.length//2 + 1
        pos = np.random.choice(pos_index, half, replace=True)
        neg = np.random.choice(neg_index, half, replace=True)
        l = np.stack([pos, neg]).T
        l = l.reshape(-1)
        l = l[:self.length]
        return iter(l)

    def __len__(self):
        return self.length

##############################################################


def run_check_train_dataset():

    dataset = XrayDataset(
        mode='train',
        csv=['train-rle.single-mask.csv', ],
        split='train1_10075.npy',
        augment=None,
    )
    print(dataset)
    # exit(0)

    for n in range(0, len(dataset)):
        i = n  # i = np.random.choice(len(dataset))

        image, mask, infor = dataset[i]
        if infor.label == 0:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        contour = draw_contour_overlay(image.copy(), mask, 2)
        mask = draw_mask(mask)

        # ----
        print('%05d : %s' % (i, infor.image_id))
        image_show('image', image, 0.5)
        image_show('contour', contour, 0.5)
        image_show('mask', mask, 1)
        cv2.waitKey(0)


def run_check_test_dataset():

    dataset = XrayDataset(
        mode='test',
        csv=['test-rle.single-mask.csv'],
        split='test_1372.npy',
        augment=None,
    )
    print(dataset)
    # exit(0)

    for n in range(0, len(dataset)):
        i = n  # i = np.random.choice(len(dataset))

        image, mask, infor = dataset[i]
        # if infor.label==0: continue
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        contour = draw_contour_overlay(image.copy(), mask, 2)
        mask = draw_mask(mask)

        # ----
        print('%05d : %s' % (i, infor.image_id))
        image_show('image', image, 0.5)
        image_show('contour', contour, 0.5)
        image_show('mask', mask, 1)
        cv2.waitKey(0)


def run_check_data_loader():

    dataset = XrayDataset(
        mode='train',
        csv=['train-rle.single-mask.csv', ],
        split='train1_10075.npy',
        augment=None,
    )
    print(dataset)
    loader = DataLoader(
        dataset,
        sampler=BalanceClassSampler(dataset),
        #sampler     = SequentialSampler(dataset),
        #sampler     = RandomSampler(dataset),
        batch_size=32,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=null_collate
    )

    for t, (input, truth, infor) in enumerate(loader):
        label = [i.label for i in infor]

        print('----t=%d---' % t)
        print('')
        print(infor)
        print(input.shape)
        print(truth.shape)
        print(label)
        print('')

        if 1:
            batch_size = len(infor)
            input = input.data.cpu().numpy()
            truth = truth.data.cpu().numpy()

            for b in range(batch_size):
                image = input[b, 0]
                mask = truth[b, 0]

                print(label[b])
                image_show_norm('mask', mask, resize=0.25)
                image_show_norm('image', image, resize=0.25)
                cv2.waitKey(0)


def run_check_augment():

    def augment(image, mask, infor):
        image, mask = do_random_scale_rotate(image, mask)
        #image = do_random_log_contast(image)
        return image, mask, infor

    dataset = XrayDataset(
        mode='train',
        csv=['train-rle.single-mask.csv', ],
        split='train1_10075.npy',
        augment=None,  # None
    )
    print(dataset)

    for t in range(len(dataset)):
        image, mask, infor = dataset[t]

        print('----t=%d---' % t)
        print('')
        print('infor\n', infor)
        print(image.shape)
        print(mask.shape)
        print('')

        if infor.label == 0:
            continue
        overlay = draw_input(image)

        image_show_norm('original_mask', mask,  resize=0.25)
        image_show('original_image', image, resize=0.25)
        image_show('original_overlay', overlay, resize=0.25)
        cv2.waitKey(1)

        if 1:
            for i in range(100):
                image1, mask1, infor1 = augment(image, mask, infor)

                overlay1 = draw_input(image1)
                overlay1 = draw_contour_overlay(overlay1, mask1, thickness=5)

                image_show_norm('mask', mask1,  resize=0.25)
                image_show('image', image1, resize=0.25)
                image_show('overlay', overlay1, resize=0.25)
                cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_test_dataset()

    # run_check_train_dataset()
    # run_check_data_loader()
    # run_check_augment()

    print('\nsucess!')
