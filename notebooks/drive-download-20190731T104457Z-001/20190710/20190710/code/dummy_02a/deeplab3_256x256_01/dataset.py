from common import *
from data import *

DATA_DIR = '/root/share/project/kaggle/2019/chest/data'


class XrayDataset(Dataset):
    def __init__(self, split, csv, folder, mode, augment=None,):

        self.split = split
        self.csv = csv
        self.folder = folder
        self.mode = mode
        self.augment = augment

        self.df = pd.read_csv(DATA_DIR + '/%s' % csv)
        self.gb = self.df.groupby('ImageId')
        self.df['count'] = self.gb['ImageId'].transform('count')
        self.df.loc[self.df['EncodedPixels'] == '-1', 'count'] = 0

        self.dicom_file = get_dicom_file(DATA_DIR + '/' + folder)
        if split is None:
            uid = list(self.gb.groups.keys())
            uid.sort()
            self.uid = uid
        else:
            uid = list(np.load(DATA_DIR + '/split/%s' %
                               split, allow_pickle=True))
            uid.sort()
            self.uid = uid

        # if 1: #<debug>
        #     self.uid = self.uid[:1000]

        if self.mode == 'train':
            num_component = []
            for i in self.uid:
                df = self.gb.get_group(i)
                num_component.append(df['count'].values[0])
            self.num_component = np.array(num_component, np.int32)
        else:
            self.num_component = np.zeros(len(self.uid), np.int32)

    def __str__(self):
        if self.mode == 'train':
            string = ''\
                + '\tmode   = %s\n' % self.mode \
                + '\tsplit  = %s\n' % self.split \
                + '\tcsv    = %s\n' % self.csv \
                + '\tfolder = %s\n' % self.folder \
                + '\tnum_component[n] = %d\n' % (sum(self.num_component >= 1)) \
                + '\tnum_component[0] = %d\n' % (sum(self.num_component == 0)) \
                + '\tlen    = %d\n' % len(self)

        if self.mode == 'test':
            string = ''\
                + '\tmode   = %s\n' % self.mode \
                + '\tsplit  = %s\n' % self.split \
                + '\tcsv    = %s\n' % self.csv \
                + '\tfolder = %s\n' % self.folder \
                + '\tnum_component[n] = ?\n' \
                + '\tnum_component[0] = ?\n' \
                + '\tlen    = %d\n' % len(self)

        return string

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, index):
        # print(index)
        i = self.uid[index]

        data = pydicom.read_file(self.dicom_file[i])
        image = data.pixel_array

        if self.mode == 'train':
            df = self.gb.get_group(i)
            component, num_component = gb_to_component(df)

        if self.mode == 'test':
            component = np.zeros((1024, 1024), np.float32)
            num_component = 0

        infor = Struct(
            image_id=i,
            index=index,
            #df = df,
            #is_copy = True,
        )

        if self.augment is None:
            return image, component, num_component, infor
        else:
            return self.augment(image, component, num_component, infor)

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


##############################################################

def null_augment(image, component, num_component, infor):
    mask = component_to_mask(component)
    return image, mask, num_component, infor


def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth = []
    num_component = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
        num_component.append(batch[b][2])
        infor.append(batch[b][3])

    input = np.stack(input).astype(np.float32)
    input = input/255
    truth = np.stack(truth).astype(np.float32)

    input = torch.from_numpy(input).unsqueeze(1).float()
    truth = torch.from_numpy(truth).unsqueeze(1).float()

    return input, truth, num_component, infor


class BalanceClassSampler(Sampler):

    def __init__(self, dataset, length=None):
        self.dataset = dataset

        if length is None:
            length = len(self.dataset)

        self.length = length

    def __iter__(self):
        pos_index = np.where(self.dataset.num_component >= 1)[0]
        neg_index = np.where(self.dataset.num_component == 0)[0]

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
        csv='train-rle.csv',
        folder='dicom/dicom-images-train',
        split='train_10075.npy',
        #split  = 'valid_600.npy',
        #split  = None,
        augment=null_augment,  # None #
    )
    print(dataset)
    # exit(0)

    for n in range(0, len(dataset)):
        i = n  # i = np.random.choice(len(dataset))

        image, mask, num_component, infor = dataset[i]
        if num_component == 0:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        contour = image.copy()
        contour = draw_contour_overlay(contour, mask, 2)
        mask = draw_mask_overlay(mask)

        # ----
        print('%05d : %s' % (i, infor.image_id))
        image_show('image', image, 0.5)
        image_show('contour', contour, 0.5)
        image_show('mask', mask, 1)
        cv2.waitKey(0)


def run_check_data_loader():

    dataset = XrayDataset(
        mode='train',
        csv='train-rle.csv',
        folder='dicom/dicom-images-train',
        split=None,
        augment=null_augment,  # None #
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

    for t, (input, truth, num_component, infor) in enumerate(loader):

        print('----t=%d---' % t)
        print('')
        print(infor)
        print(input.shape)
        print(truth.shape)
        print(num_component)
        print('')

        if 1:
            batch_size = len(infor)
            input = input.data.cpu().numpy()
            truth = truth.data.cpu().numpy()

            for b in range(batch_size):
                image = input[b, 0]
                mask = truth[b, 0]

                print(num_component[b])
                image_show_norm('mask', mask, resize=0.25)
                image_show_norm('image', image, resize=0.25)
                cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_train_dataset()
    # run_check_data_loader()

    print('\nsucess!')
