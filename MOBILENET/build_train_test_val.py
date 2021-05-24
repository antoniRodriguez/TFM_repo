import os
from threading import main_thread
import xml.etree.cElementTree as ET
from matplotlib import pyplot as plt
from xml.dom import minidom
from tqdm import tqdm
import cv2


BASE_PATH = '/home/group00/working/Antoni_THESIS/DATASETS/CCPD2019/'
DATASET_PATH = BASE_PATH + '/ccpd_base'

TRAIN_FOLDER = '/home/group00/working/Antoni_THESIS/MOBILENET/train_full'
TEST_FOLDER = '/home/group00/working/Antoni_THESIS/MOBILENET/test_full'
TRAIN_SIZE = 5000
TEST_SIZE = 1000

def get_image_information(path_to_image):
    '''
    image information: bbox coordinates
     > everything that needs to go on the xml annotation file
    '''
    iname = path_to_image.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    [[x1,y1],[x2,y2]] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]

    return [x1,y1,x2,y2]


def generate_xml_from_filename(filename,new_filename,split):
    original_folder = filename.rsplit('/', 1)[0] # 'ccpd_base'
    filename = filename.rsplit('/', 1)[-1] # '123412341234.jpg'

    image_path = os.path.join('/home/group00/working/Antoni_THESIS/MOBILENET',split+'_full',new_filename+'.jpg') # '/home/group00/working/Antoni_THESIS/MOBILENET/train/.jpg\n'
    image_folder = split+'_full'
    iname_split = filename.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    [[xmin,ymin],[xmax,ymax]] = [[int(eel) for eel in el.split('&')] for el in iname_split[2].split('_')]
    img = cv2.cvtColor(cv2.imread(os.path.join(BASE_PATH,original_folder,filename[:-1])), cv2.COLOR_BGR2RGB)
    [width,height,depth] = img.shape
    

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = image_folder
    ET.SubElement(annotation, "filename").text = new_filename+'.jpg'
    ET.SubElement(annotation, "path").text = image_path
    ET.SubElement(annotation, "original_path").text = os.path.join(BASE_PATH,original_folder,filename[:-1])


    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = 'Unknown'

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = '0'

    # OBJECT
    object = ET.SubElement(annotation, "object")
    ET.SubElement(object, "name").text = 'LicensePlate'
    ET.SubElement(object, "pose").text = 'Unspecified'
    ET.SubElement(object, "truncated").text = '0'
    ET.SubElement(object, "difficult").text = '0'

    bbox = ET.SubElement(object, "bndbox")
    ET.SubElement(bbox, "xmin").text = str(xmin)
    ET.SubElement(bbox, "ymin").text = str(ymin)
    ET.SubElement(bbox, "xmax").text = str(xmax)
    ET.SubElement(bbox, "ymax").text = str(ymax)


    xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")

    # save xml
    with open(os.path.join('/home/group00/working/Antoni_THESIS/MOBILENET/RealTimeObjectDetection/Tensorflow/workspace/images',split+'_full',new_filename)+".xml", "w") as f:
        f.write(xmlstr)
    
    # save image
    cv2.imwrite(str(os.path.join('/home/group00/working/Antoni_THESIS/MOBILENET/RealTimeObjectDetection/Tensorflow/workspace/images',split+'_full',new_filename+'.jpg')),img)
    
    





def build_img_annotations_folder(split):
    print('Building '+ split + ' split...')
    # max_size = 0
    # if split == 'train':
    #     max_size = TRAIN_SIZE
    # else:
    #     max_size = TEST_SIZE

    with open(BASE_PATH+"/splits/"+split+".txt", "r") as f:
        #for idx,filename in tqdm(enumerate(f)):
        for idx,filename in tqdm(enumerate(f)): # filename: 'ccpd_base/009281609195... .jpg\n'
            # if idx == max_size: break
            
            # print(split+' || '+str(idx+1)+'/'+str(max_size)+'...')
            generate_xml_from_filename(filename, new_filename=str(idx).zfill(7),split=split)
            
            





if __name__ == '__main__':
    build_img_annotations_folder('train')
    build_img_annotations_folder('test')
    build_img_annotations_folder('val')