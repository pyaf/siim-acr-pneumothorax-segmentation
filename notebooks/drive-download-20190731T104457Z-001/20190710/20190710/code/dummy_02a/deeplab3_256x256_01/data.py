from common import *
import pydicom

# https://www.kaggle.com/adkarhe/dicom-images
# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview/resources
# https://www.kaggle.com/abhishek/train-your-own-mask-rcnn


# https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/98478#latest-568453
# Dataset Update: Non-annotated instances/images
# def run_fix_kaggle_data_error():
#     csv_file    = '/root/share/project/kaggle/2019/chest/data/__download__/train-rle_old.csv'
#     remove_file = '/root/share/project/kaggle/2019/chest/data/__download__/non-diagnostic-train'
#     dicom_dir   = '/root/share/project/kaggle/2019/chest/data/dicom/dicom-images-train'
#
#
#     # csv_file    = '/root/share/project/kaggle/2019/chest/data/__download__/test-rle_old.csv'
#     # remove_file = '/root/share/project/kaggle/2019/chest/data/__download__/non-diagnostic-test'
#     # dicom_dir   = '/root/share/project/kaggle/2019/chest/data/dicom/dicom-images-test'
#     #---
#     dicom_file = get_dicom_file(dicom_dir)
#     dicom_id = set(dicom_file.keys())
#
#     df = pd.read_csv(csv_file)
#     df_id  = set(df.ImageId.values)
#
#     remove_id = []
#     non_diagnostic = read_list_from_file(remove_file)
#     for k,v in dicom_file.items():
#         #print(k,v)
#         for s in non_diagnostic:
#             if s in v:
#                 print (v)
#                 remove_id.append(k)
#
#     remove_id=set(remove_id)
#
#     #----
#     print('remove_id  :',len(remove_id))
#     print('df_id      :',len(df_id))
#     print('dicom_id   :',len(dicom_id))
#     print('')
#     print('dicom_id   ∩  df_id     :',len(set(dicom_id).intersection(df_id)))
#     print('dicom_id   ∩  remove_id :',len(set(dicom_id).intersection(remove_id)))
#     print('df_id      ∩  remove_id :',len(set(df_id).intersection(remove_id)))
#     exit(0)


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
    if np.all(rle == '-1'):
        component = np.zeros((1, height, width), np.float32)
        return component,  0

    component = np.array([run_length_decode(r, height, width, 1) for r in rle])
    num_component = len(component)

    return component, num_component


def component_to_mask(component):
    mask = component.sum(0)
    mask = (mask > 0.5).astype(np.float32)
    return mask


def mask_to_component(mask, threshold=0.5):
    H, W = mask.shape
    binary = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, label = cv2.connectedComponents(binary.astype(np.uint8))

    num_component = num_component-1
    component = np.zeros((num_component, H, W), np.float32)
    for i in range(0, num_component):
        #component[i][label==(i+1)] = mask[label==(i+1)]
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

    input = draw_input_overlay(input)
    input1 = cv2.resize(input, dsize=(512, 512))

    if truth.shape != (512, 512):
        truth1 = cv2.resize(truth, dsize=(512, 512))
        probability1 = cv2.resize(probability, dsize=(512, 512))
    else:
        truth1 = truth
        probability1 = probability

    # ---
    overlay1 = draw_truth_overlay(input1.copy(), truth1,   0.5)
    overlay2 = draw_predict_overlay(input1.copy(), probability1, 0.5)

    overlay3 = np.zeros((512, 512, 3), np.uint8)
    overlay3 = draw_truth_overlay(overlay3, truth1, 1.0)
    overlay3 = draw_predict_overlay(overlay3, probability1, 1.0)
    draw_shadow_text(overlay3, 'truth', (2, 12),  0.5, (0, 0, 255), 1)
    draw_shadow_text(overlay3, 'predict', (2, 24),  0.5, (0, 255, 0), 1)

    # <todo> results afer post process ...
    overlay4 = np.zeros((512, 512, 3), np.uint8)
    overlay = np.hstack([
        input,
        np.hstack([
            np.vstack([overlay1, overlay2]),
            np.vstack([overlay4, overlay3]),
        ])
    ])
    return overlay


### check #######################################################################################

# lstrip
def run_process_0():

    df = pd.read_csv(
        '/root/share/project/kaggle/2019/chest/data/train-rle.csv')
    df.rename(columns={' EncodedPixels': 'EncodedPixels', }, inplace=True)
    df['EncodedPixels'] = df['EncodedPixels'].str.lstrip(to_strip=None)

    df.to_csv('/root/share/project/kaggle/2019/chest/data/train-rle.csv',
              columns=['ImageId', 'EncodedPixels'], index=False)

    zz = 0


def run_split_dataset():

    df = pd.read_csv(
        '/root/share/project/kaggle/2019/chest/data/train-rle.more.csv')
    gb = df.groupby('ImageId')

    uid = list(gb.groups.keys())

    num_component = []
    for i in uid:
        df = gb.get_group(i)
        num_component.append(df['count'].values[0])  # count= num of instances
    num_component = np.array(num_component, np.int32)

    neg_index = np.where(num_component == 0)[0]
    pos_index = np.where(num_component >= 1)[0]  # those which have more than one instances

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


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_split_dataset()

    print('\nsucess!')
