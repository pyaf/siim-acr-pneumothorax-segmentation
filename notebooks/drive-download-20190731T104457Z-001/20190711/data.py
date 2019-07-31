from common import *

# https://www.kaggle.com/adkarhe/dicom-images
# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview/resources
# https://www.kaggle.com/abhishek/train-your-own-mask-rcnn

# ------
# component
# components
# mask
# ------


import pydicom

# initial version of kaggle data processing


# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/98478#latest-568453
# Dataset Update: Non-annotated instances/images
def run_fix_kaggle_data_error():
    csv_file = '/root/share/project/kaggle/2019/chest/data/__download__/train-rle_old.csv'
    remove_file = '/root/share/project/kaggle/2019/chest/data/__download__/non-diagnostic-train'
    dicom_dir = '/root/share/project/kaggle/2019/chest/data/dicom/dicom-images-train'

    # csv_file    = '/root/share/project/kaggle/2019/chest/data/__download__/test-rle_old.csv'
    # remove_file = '/root/share/project/kaggle/2019/chest/data/__download__/non-diagnostic-test'
    # dicom_dir   = '/root/share/project/kaggle/2019/chest/data/dicom/dicom-images-test'
    # ---
    dicom_file = get_dicom_file(dicom_dir)
    dicom_id = set(dicom_file.keys())

    df = pd.read_csv(csv_file)
    df_id = set(df.ImageId.values)

    remove_id = []
    non_diagnostic = read_list_from_file(remove_file)
    for k, v in dicom_file.items():
        # print(k,v)
        for s in non_diagnostic:
            if s in v:
                print(v)
                remove_id.append(k)

    remove_id = set(remove_id)

    # ----
    print('remove_id  :', len(remove_id))
    print('df_id      :', len(df_id))
    print('dicom_id   :', len(dicom_id))
    print('')
    print('dicom_id   ∩  df_id     :', len(set(dicom_id).intersection(df_id)))
    print('dicom_id   ∩  remove_id :', len(
        set(dicom_id).intersection(remove_id)))
    print('df_id      ∩  remove_id :', len(set(df_id).intersection(remove_id)))
    exit(0)


'''
You should be expecting 10712 images in the train set 
and 1377 images in the public test set.


for test *.dcm files:
 
remove_id  : 4
df_id      : 1372
dicom_id   : 1377

dicom_id   ∩  df_id     : 1372
dicom_id   ∩  remove_id : 4
df_id      ∩  remove_id : 0


for train *.dcm files:

remove_id  : 33
df_id      : 10675
dicom_id   : 10712

dicom_id   ∩  df_id     : 10675
dicom_id   ∩  remove_id : 33
df_id      ∩  remove_id : 0


'''


# ----
def get_dicom_file(folder):
    dicom_file = glob.glob(folder + '/**/**/*.dcm')
    dicom_file = sorted(dicom_file)
    image_id = [f.split('/')[-1][:-4] for f in dicom_file]
    dicom_file = dict(zip(image_id, dicom_file))
    return dicom_file


# ----
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


def gb_to_component(df, height=1024, width=1024):

    rle = df['EncodedPixels'].values
    if rle[0] == '-1':
        component = np.zeros((1, height, width), np.float32)
        return component,  0

    #box = df[['x0','y0','x1','y1']].values
    component = np.array([run_length_decode(r, height, width, 1) for r in rle])
    num_component = len(component)

    return component, num_component


def component_to_mask(component):
    mask = component.sum(0)
    mask = (mask > 0.5).astype(np.float32)
    return mask


def mask_to_component(mask, threshold=0.5):
    H, W = mask.shape
    mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, label = cv2.connectedComponents(mask.astype(np.uint8))

    num_component = num_component-1
    component = np.zeros((num_component, H, W), np.float32)
    for i in range(0, num_component):
        component[i] = label == (i+1)

    return component, num_component


### draw ############################################

def draw_input_overlay(image):
    overlay = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    return overlay


def draw_mask_overlay(mask):
    height, width = mask.shape
    overlay = np.zeros((height, width, 3), np.uint8)
    overlay[mask > 0] = (0, 0, 255)
    return overlay


def draw_truth_overlay(image, component, alpha=0.5):
    component = component*255
    overlay = image.astype(np.float32)
    overlay[:, :, 2] += component*alpha
    overlay = np.clip(overlay, 0, 255)
    overlay = overlay.astype(np.uint8)
    return overlay


def draw_predict_overlay(image, component, alpha=0.5):
    component = component*255
    overlay = image.astype(np.float32)
    overlay[:, :, 1] += component*alpha
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


def draw_contour_overlay(image, component, thickness=1):
    contour = mask_to_inner_contour(component)
    if thickness == 1:
        image[contour] = (0, 0, 255)
    else:
        for y, x in np.stack(np.where(contour)).T:
            cv2.circle(image, (x, y), thickness,
                       (0, 0, 255), lineType=cv2.LINE_4)
    return image


###- combined results --#
def draw_result_overlay(input, truth, probability):
    H, W = input.shape
    h, w = H//2, W//2

    input = draw_input_overlay(input)
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


### check #######################################################################################

def run_check_length_encode0():

    dicom_file = get_dicom_file(
        '/root/share/project/kaggle/2019/chest/data/dicom/dicom-images-train')

    #df = pd.read_csv('/root/share/project/kaggle/2019/chest/data/debug-rle.csv')
    df = pd.read_csv(
        '/root/share/project/kaggle/2019/chest/data/train-rle.csv')
    gb = df.groupby('ImageId')

    uid = list(gb.groups.keys())
    for i in uid:

        data = pydicom.read_file(dicom_file[i])
        image = data.pixel_array
        df = gb.get_group(i)

        components, num_component = gb_to_components(df)
        mask = components_to_mask(components)

        if num_component < 1:
            continue

        image = draw_overlay(image)
        for c in range(num_component):
            components[c] = draw_truth_overlay(image, components[c], alpha=0.2)

        mask = draw_mask_overlay(mask)

        #print('%d, %s'%(num_component,i))
        # image_show('overlay',overlay,0.25)
        image_show('image', image, 0.25)
        # image_show_norm('mask',mask,resize=0.25)
        image_show('mask', mask, resize=0.25)
        image_show_norm('components', np.hstack(components), resize=0.25)

        # component = split_mask_to_component(mask)
        # for t,c in enumerate(component):
        #     image_show('c-%d'%t,c,0.25)
        #     run_length_encode(c)

        cv2.waitKey(0)


def run_check_length_encode1():

    df0 = pd.read_csv(
        '/root/share/project/kaggle/2019/chest/data/debug-rle.more.csv')
    gb0 = df0.groupby('ImageId')

    image_id = []
    encoded_pixel = []

    uid = list(gb0.groups.keys())
    for i in uid:
        df = gb0.get_group(i)
        components0, num_component0 = gb_to_component(df)
        mask = components_to_mask(components0)

        # -----
        components1, num_component1 = mask_to_components(mask)
        if num_component1 == 0:
            image_id.append(i)
            encoded_pixel.append('-1')
        else:
            for component in components1:
                r = run_length_encode(component)
                image_id.append(i)
                encoded_pixel.append(r)

        print(i)
        print(num_component0, num_component1)
        print(df['EncodedPixels'].values[-1])
        print(encoded_pixel[-1])
        print('')

    df1 = pd.DataFrame(list(zip(image_id, encoded_pixel)),
                       columns=['ImageId', 'EncodedPixels'])

    df0 = df0.sort_values(
        by=['ImageId', 'EncodedPixels'], ascending=[True, True])
    df1 = df1.sort_values(
        by=['ImageId', 'EncodedPixels'], ascending=[True, True])
    df0.reset_index(drop=True, inplace=True)
    df1.reset_index(drop=True, inplace=True)

    print('df0\n', df0.head(20))
    print('df1\n', df1.head(20))
    print('')
    print(df0.equals(df1))
    # print(df0.values[14])
    # print(df1.values[14])

# lstrip


def run_process_0():

    df = pd.read_csv(
        '/root/share/project/kaggle/2019/chest/data/train-rle.csv')
    df.rename(columns={' EncodedPixels': 'EncodedPixels', }, inplace=True)
    df['EncodedPixels'] = df['EncodedPixels'].str.lstrip(to_strip=None)

    df.to_csv('/root/share/project/kaggle/2019/chest/data/train-rle.csv',
              columns=['ImageId', 'EncodedPixels'], index=False)

    zz = 0


def run_process_1():

    #df = pd.read_csv('/root/share/project/kaggle/2019/chest/data/debug-rle.csv')
    df = pd.read_csv(
        '/root/share/project/kaggle/2019/chest/data/train-rle.csv')
    gb = df.groupby('ImageId')
    count = gb.agg('count')

    num = []
    for image_id, encoded_pixel in df.values:
        if encoded_pixel == '-1':
            num.append(0)
        else:
            num.append(count.loc[image_id].values[0])

    df['num'] = num

    df.to_csv('/root/share/project/kaggle/2019/chest/data/train-rle.more.csv',
              columns=['ImageId', 'count', 'EncodedPixels'], index=False)
    df[:1000].to_csv('/root/share/project/kaggle/2019/chest/data/debug-rle.more.csv',
                     columns=['ImageId', 'count', 'EncodedPixels'], index=False)
    zz = 0


def run_process_2():

    dicom_file = get_dicom_file(
        '/root/share/project/kaggle/2019/chest/data/dicom/dicom-images-train')

    #df = pd.read_csv('/root/share/project/kaggle/2019/chest/data/debug-rle.csv')
    df = pd.read_csv(
        '/root/share/project/kaggle/2019/chest/data/train-rle.csv')
    gb = df.groupby('ImageId')
    gb_count = gb.agg('count')

    df['x0'] = 0
    df['y0'] = 0
    df['x1'] = 0
    df['y1'] = 0
    df['count'] = 0
    for t, v in enumerate(df.values):
        image_id, encoded_pixel = v[:2]
        if encoded_pixel == '-1':
            pass
        else:
            df.iloc[t, df.columns.get_loc(
                'count')] = gb_count.loc[image_id].values[0]

            rle = encoded_pixel
            component = run_length_decode(
                rle, height=1024, width=1024, fill_value=1)

            cc = (component > 0.5).astype(np.float32)
            yy = np.any(cc > 0.5, axis=1)
            xx = np.any(cc > 0.5, axis=0)
            x0, x1 = np.where(xx)[0][[0, -1]]
            y0, y1 = np.where(yy)[0][[0, -1]]
            x1 += 1
            y1 += 1
            print(x0, x1, y0, y1)

            df.iloc[t, df.columns.get_loc('x0')] = x0
            df.iloc[t, df.columns.get_loc('y0')] = y0
            df.iloc[t, df.columns.get_loc('x1')] = x1
            df.iloc[t, df.columns.get_loc('y1')] = y1

            cc[y0, x0:x1-1] = 0.5
            cc[y1-1, x0:x1-1] = 0.5
            image_show_norm('cc', cc, resize=1)
            cv2.waitKey(1)

        print(image_id)

    # ----

    column = ['ImageId', 'count', 'x0', 'y0', 'x1', 'y1', 'EncodedPixels']
    df.to_csv('/root/share/project/kaggle/2019/chest/data/train-rle.more.csv',
              columns=column, index=False)
    df[:1000].to_csv('/root/share/project/kaggle/2019/chest/data/debug-rle.more.csv',
                     columns=column, index=False)
    zz = 0


def run_split_dataset():

    df = pd.read_csv(
        '/root/share/project/kaggle/2019/chest/data/train-rle.csv')
    gb = df.groupby('ImageId')

    uid = list(gb.groups.keys())

    num_component = []
    for i in uid:
        df = gb.get_group(i)
        num_component.append(df['count'].values[0])
    num_component = np.array(num_component, np.int32)

    neg_index = np.where(num_component == 0)[0]
    pos_index = np.where(num_component >= 1)[0]

    print('num_component==0 :  %d' % (len(neg_index)))
    print('num_component>=1 :  %d' % (len(pos_index)))
    print('len(uid) :  %d' % (len(uid)))

    np.random.shuffle(neg_index)
    np.random.shuffle(pos_index)

    train_split = np.concatenate([neg_index[300:], pos_index[300:], ])
    valid_split = np.concatenate([neg_index[:300], pos_index[:300], ])

    uid = np.array(uid, np.object)
    train_split = uid[train_split]
    valid_split = uid[valid_split]

    np.save('/root/share/project/kaggle/2019/chest/data/split/train_%d' %
            len(train_split), train_split)
    np.save('/root/share/project/kaggle/2019/chest/data/split/valid_%d' %
            len(valid_split), valid_split)

    zz = 0


def run_split_dataset1():

    df = pd.read_csv(
        '/root/share/project/kaggle/2019/chest/data/train-rle.csv')
    gb = df.groupby('ImageId')
    df['count'] = gb['ImageId'].transform('count')
    df.loc[df['EncodedPixels'] == '-1', 'count'] = 0

    image_id = list(gb.groups.keys())

    num_component = []
    for i in image_id:
        d = gb.get_group(i)
        num_component.append(d['count'].values[0])
    num_component = np.array(num_component, np.int32)

    neg_index = np.where(num_component == 0)[0]
    pos_index = np.where(num_component >= 1)[0]

    print('num_component==0 :  %d' % (len(neg_index)))
    print('num_component>=1 :  %d' % (len(pos_index)))
    print('len(image_id)    :  %d' % (len(image_id)))

    np.random.shuffle(neg_index)
    np.random.shuffle(pos_index)

    #neg_split = np.array_split(neg_index,8)
    #pos_split = np.array_split(pos_index,8)
    # >>> 8296/8
    # 1037.0
    # >>> 2379/8
    # 297.375

    neg_split = []
    pos_split = []
    S = 6
    for s in range(S):
        if s == S-1:
            neg_split.append(neg_index[s*300:])
            pos_split.append(pos_index[s*300:])
        else:
            neg_split.append(neg_index[s*300:(s+1)*300])
            pos_split.append(pos_index[s*300:(s+1)*300])

    image_id = np.array(image_id, np.object)
    for s in range(S):
        valid_split = np.concatenate([neg_split[s], pos_split[s], ])

        train_split = []
        for t in range(S):
            if t != s:
                train_split.extend(list(neg_split[t]))
                train_split.extend(list(pos_split[t]))

        train_split = np.array(train_split, np.int32)
        valid_split = np.array(valid_split, np.int32)
        train_split = image_id[train_split]
        valid_split = image_id[valid_split]

        np.save('/root/share/project/kaggle/2019/chest/data/split/train%d_%d.npy' % (
            s, len(train_split)), train_split)
        np.save('/root/share/project/kaggle/2019/chest/data/split/valid%d_%d.npy' % (
            s, len(valid_split)), valid_split)
    #
    zz = 0


def run_split_check():
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


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # run_fix_kaggle_data_error()
    run_check_length_encode0()
    # run_check_length_encode1()
    # run_process_2()

    # run_split_dataset()
    # run_split_dataset1()
    # run_split_check()

    print('\nsucess!')
