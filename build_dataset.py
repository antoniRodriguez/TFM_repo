import pandas as pd
from tqdm import tqdm
import cv2
import os


def extract_box(filename):
    '''
    image filename example:
        "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg".
        >       ...[154&383_386&473]...
        
        bbox: [154, 383, 386, 473] ==> (x1,y1,x2,y2) top-left and bottom-right
    '''
    iname = filename.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    [[x1,y1],[x2,y2]] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]

    return [x1,y1,x2,y2]

def format_bbox_coordinates(dataset, im_path):
    image = cv2.imread(im_path)
    w,h = image.shape
    dataset["x_min"] /= h
    dataset["y_min"] /= w
    dataset["x_max"] /= h
    dataset["y_max"] /= w

# CREATE DATASET DATAFRAME WITH FILENAME AND BBOX ANNOTATIONS
def load_dataset(base_path, isTrain = True):
    max_n_samples = 999
    dataset = dict()
    dataset["image_name"] = list()
    dataset["x_min"] = list()
    dataset["y_min"] = list()
    dataset["x_max"] = list()
    dataset["y_max"] = list()
    dataset["class_name"] = list()
    if isTrain:
        split = "train"
    else:
        split = "val"
    print(base_path+"splits/"+split+".txt")
    with open(base_path+"/splits/"+split+".txt", "r") as f:
        for idx,filename in tqdm(enumerate(f)):
            if len(filename) == 0: break
            filename = filename.strip()
            [x1,y1,x2,y2] = extract_box(filename)

            dataset["image_name"].append(os.path.join(base_path,filename))
            dataset["x_min"] = x1
            dataset["y_min"] = y1
            dataset["x_max"] = x2
            dataset["y_max"] = y2
            format_bbox_coordinates(dataset, os.path.join(base_path,filename))
            dataset["class_name"] = "License_Plate"
            if idx == max_n_samples:
                break

            #im = cv2.imread(os.path.join(base_path,filename)) # cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
            #cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),2)
            #cv2.putText(im,'License Plate',(x2+10,y2),0,0.3,(0,255,0))
            #cv2.imwrite('ccpd_'+str(idx)+'.jpg', im)

    df = pd.DataFrame(dataset)
    return df



def create_csv_files(base_path):
    '''
    if os.path.isfile('Antoni_THESIS/annotations.csv') and os.path.isfile('Antoni_THESIS/classes.csv'):
        print("train.csv ALREADY EXISTS!")
        print("classes.csv ALREADY EXISTS!")
    '''

    #base_path = "/home/group00/working/Antoni_THESIS/DATASETS/CCPD2019/"
    #train_df = load_dataset(base_path=base_path, isTrain=True)
    print("Loading dataset...")
    test_df = load_dataset(base_path=base_path, isTrain=False)
    print("Dataset loaded!")


    # save csv files for training
    ANNOTATIONS_FILE_TRAIN = '/home/group00/working/Antoni_THESIS/train.csv'
    ANNOTATIONS_FILE_TEST = '/home/group00/working/Antoni_THESIS/test.csv'
    CLASSES_FILE = '/home/group00/working/Antoni_THESIS/classes.csv'

    #  annotations
    #train_df.to_csv(ANNOTATIONS_FILE_TRAIN, index=False, header=None)
    print("Writting csv file...")
    test_df.to_csv(ANNOTATIONS_FILE_TEST, index=False, header=None)
    print("csv file saved!")

    # classes
    # classes = set(['License_Plate'])
    # with open(CLASSES_FILE, 'w') as f:
    #     for i,line in enumerate(sorted(classes)):
    #         f.write('{}, {}'.format(line,i))