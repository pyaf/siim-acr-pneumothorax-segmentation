from common import *
import pydicom

#### etc ###########################################################


def get_dicom_file(folder):
    dicom_file = glob.glob(folder + '/**/**/*.dcm')
    dicom_file = sorted(dicom_file)
    image_id = [f.split('/')[-1][:-4] for f in dicom_file]
    dicom_file = dict(zip(image_id, dicom_file))
    return dicom_file


#### kaggle score ###########################################################

# https://www.kaggle.com/mnpinto/pneumothorax-fastai-starter-u-net-128x128
def run_length_decode(rle, height=1024, width=1024, fill_value=1):

    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.split(' ')])
    rle = rle.reshape(-1, 2)

    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end

    component = component.reshape(width, height).T
    return component


# 1.2.276.0.7230010.3.1.4.8323329.10005.1517875220.958951
#   209126 1 1019 6 1015 10 1012 13 1010 14 1008 16 1007 16 1006 18 1004 20 1003 20 1002 22
def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0]+1
    end = np.where(component[:-1] > component[1:])[0]+1
    length = end-start

    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i]-end[i-1], length[i]])

    rle = ' '.join([str(r) for r in rle])
    return rle


def kaggle_metric_one(predict, truth):

    if truth.sum() == 0:
        if predict.sum() == 0:
            return 1
        else:
            return 0

    # ----
    predict = predict.reshape(-1)
    truth = truth.reshape(-1)

    intersect = predict*truth
    union = predict+truth
    dice = 2.0*intersect.sum()/union.sum()
    return dice

#### draw ###########################################################


def draw_input(image):
    overlay = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    return overlay


def draw_mask(mask, color=(0, 0, 255)):
    height, width = mask.shape
    overlay = np.zeros((height, width, 3), np.uint8)
    overlay[mask > 0] = color
    return overlay


def draw_truth_overlay(image, mask, alpha=0.5):
    mask = mask*255
    overlay = image.astype(np.float32)
    overlay[:, :, 2] += mask*alpha
    overlay = np.clip(overlay, 0, 255)
    overlay = overlay.astype(np.uint8)
    return overlay


def draw_predict_overlay(image, mask, alpha=0.5):
    mask = mask*255
    overlay = image.astype(np.float32)
    overlay[:, :, 1] += mask*alpha
    overlay = np.clip(overlay, 0, 255)
    overlay = overlay.astype(np.uint8)
    return overlay


# ------

def mask_to_inner_contour(component):
    component = component > 0.5
    pad = np.lib.pad(component, ((1, 1), (1, 1)), 'reflect')
    contour = component & (
        (pad[1:-1, 1:-1] != pad[:-2, 1:-1])
        | (pad[1:-1, 1:-1] != pad[2:, 1:-1])
        | (pad[1:-1, 1:-1] != pad[1:-1, :-2])
        | (pad[1:-1, 1:-1] != pad[1:-1, 2:])
    )
    return contour


def draw_contour_overlay(image, mask, thickness=1):
    contour = mask_to_inner_contour(mask)
    if thickness == 1:
        image[contour] = (0, 0, 255)
    else:
        for y, x in np.stack(np.where(contour)).T:
            cv2.circle(image, (x, y), thickness,
                       (0, 0, 255), lineType=cv2.LINE_4)
    return image


##- combined results --#
def draw_result_overlay(input, truth, probability):
    H, W = input.shape
    h, w = H//2, W//2

    input = draw_input(input)
    input1 = cv2.resize(input, dsize=(h, w))

    if truth.shape != (h, w):
        truth1 = cv2.resize(truth, dsize=(h, w))
        probability1 = cv2.resize(probability, dsize=(h, w))
    else:
        truth1 = truth
        probability1 = probability

    # ---
    overlay1 = draw_truth_overlay(input1.copy(), truth1,   0.5)
    overlay2 = draw_predict_overlay(input1.copy(), probability1, 0.5)

    overlay3 = np.zeros((h, w, 3), np.uint8)
    overlay3 = draw_truth_overlay(overlay3, truth1, 1.0)
    overlay3 = draw_predict_overlay(overlay3, probability1, 1.0)
    draw_shadow_text(overlay3, 'truth', (2, 12),  0.5, (0, 0, 255), 1)
    draw_shadow_text(overlay3, 'predict', (2, 24),  0.5, (0, 255, 0), 1)

    # <todo> results afer post process ...
    overlay4 = np.zeros((h, w, 3), np.uint8)
    overlay = np.hstack([
        input,
        np.hstack([
            np.vstack([overlay1, overlay2]),
            np.vstack([overlay4, overlay3]),
        ])
    ])
    return overlay


##### check #########################################################
def run_dicom_to_png():

    test_dicom_file = get_dicom_file(
        '/root/share/project/kaggle/2019/chest/data/dicom/dicom-images-test')
    train_dicom_file = get_dicom_file(
        '/root/share/project/kaggle/2019/chest/data/dicom/dicom-images-train')

    png_dir = '/root/share/project/kaggle/2019/chest/data'

    if 0:
        for i in list(test_dicom_file.keys()):
            print(i)
            data = pydicom.read_file(test_dicom_file[i])
            image = data.pixel_array
            cv2.imwrite(
                '/root/share/project/kaggle/2019/chest/data/png/test/%s.png' % i, image)

    if 1:
        for i in list(train_dicom_file.keys()):
            print(i)
            data = pydicom.read_file(train_dicom_file[i])
            image = data.pixel_array
            cv2.imwrite(
                '/root/share/project/kaggle/2019/chest/data/png/train/%s.png' % i, image)

    if 0:
        test_dicom_file = set(test_dicom_file.keys())
        train_dicom_file = set(train_dicom_file.keys())
        print(len(test_dicom_file))
        print(len(train_dicom_file))
        print(test_dicom_file.intersection(train_dicom_file))

        # 1377
        # 10712
        # set()


def run_check_rle():

    csv_file = \
        '/root/share/project/kaggle/2019/chest/data/train-rle.csv'

    df = pd.read_csv(csv_file)
    gb = df.groupby('ImageId')
    df['count'] = gb['ImageId'].transform('count')
    df.loc[df['EncodedPixels'] == '-1', 'count'] = 0

    image_id = list(gb.groups.keys())
    num_component = []
    for i in image_id:
        d = gb.get_group(i)
        num_component.append(d['count'].values[0])

    # ----
    for t, i in enumerate(image_id):
        if num_component[t] == 1:

            d = gb.get_group(i)
            rle = d['EncodedPixels'].values[0]

            mask = run_length_decode(
                rle, height=1024, width=1024, fill_value=1)
            r = run_length_encode(mask)
            assert(r == rle)

            print(i)
            print(rle)
            print(r)
            print('')


def run_convert_single_mask_to_cvs():

    csv_file = \
        '/root/share/project/kaggle/2019/chest/data/train-rle.csv'
    single_file = \
        '/root/share/project/kaggle/2019/chest/data/train-rle.single-mask.csv'

    df = pd.read_csv(csv_file)
    gb = df.groupby('ImageId')
    df['count'] = gb['ImageId'].transform('count')
    df.loc[df['EncodedPixels'] == '-1', 'count'] = 0

    image_id = list(gb.groups.keys())
    encoded_pixel = []

    for i in image_id:
        print(i)

        d = gb.get_group(i)
        num_component = d['count'].values[0]
        if num_component == 0:
            encoded_pixel.append('-1')

        else:
            rle = d['EncodedPixels'].values
            mask = np.array([run_length_decode(r, 1024, 1024, 1) for r in rle])
            mask = mask.sum(0)
            mask = (mask > 0.5).astype(np.float32)

            r = run_length_encode(mask)
            encoded_pixel.append(r)

    # ---
    df = pd.DataFrame(list(zip(image_id, encoded_pixel)),
                      columns=['ImageId', 'EncodedPixels'])
    df.to_csv(single_file, columns=['ImageId', 'EncodedPixels'], index=False)


def run_check_rle1():

    csv_file = \
        '/root/share/project/kaggle/2019/chest/data/train-rle.single-mask.csv'

    df = pd.read_csv(csv_file)

    image_id = df['ImageId'].values
    encoded_pixel = df['EncodedPixels'].values

    # ----
    for t, i in enumerate(image_id):
        rle = encoded_pixel[t]
        if rle == '-1':
            continue

        mask = run_length_decode(rle, height=1024, width=1024, fill_value=1)
        r = run_length_encode(mask)
        assert(r == rle)

        print(i)
        print(rle)
        print(r)
        print('')

        image_show_norm('mask', mask, resize=0.5)
        cv2.waitKey(0)


def run_make_split():
    train_split = []
    valid_split = []

    S = 5
    for s in range(S):
        t = np.load(
            '/root/share/project/kaggle/2019/chest/data/split/train%d_10075.npy' % s, allow_pickle=True)
        v = np.load(
            '/root/share/project/kaggle/2019/chest/data/split/valid%d_600.npy' % s, allow_pickle=True)
        train_split.append(t)
        valid_split.append(v)

    for i in range(S):
        t = set(train_split[i])
        v = set(valid_split[i])

        print('---- %d ----' % i)
        print('train[%d]/valid[%d] : ' % (i, i), len(t.intersection(v)))

        for j in range(S):
            if i == j:
                continue

            w = set(valid_split[j])
            print('valid[%d]/valid[%d] : ' % (i, j), w.intersection(v))

        print('')


def run_make_split1():
    df = pd.read_csv('/root/share/project/kaggle/2019/chest/data/test-rle.csv')

    image_id = np.unique(df.ImageId.values)
    encoded_pixel = ['-1']*len(image_id)

    df = pd.DataFrame(list(zip(image_id, encoded_pixel)),
                      columns=['ImageId', 'EncodedPixels'])
    df.to_csv('/root/share/project/kaggle/2019/chest/data/test-rle.single-mask.csv',
              columns=['ImageId', 'EncodedPixels'], index=False)

    t = 'test/'+image_id
    np.save('/root/share/project/kaggle/2019/chest/data/split/test_1372.npy', t)
    zz = 0


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # run_dicom_to_png()
    # run_check_rle()
    # run_check_rle1()

    # run_convert_single_mask_to_cvs()

    # run_make_split()
    run_make_split1()
