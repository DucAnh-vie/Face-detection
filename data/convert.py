# We take it from yolo face v2

import shutil
import os
import xml.etree.ElementTree as ET
from skimage import io
from os import getcwd

# Class definition
classes = ["face"]

headstr = """\
<annotation>
    <folder>VOC2012</folder>
    <filename>%06d.jpg</filename>
    <source>
        <database>My Database</database>
        <annotation>PASCAL VOC2012</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''


def writexml(idx, head, bbxes, tail):
    filename = ("Annotations/%06d.xml" % (idx))
    with open(filename, "w") as f:
        f.write(head)
        for bbx in bbxes:
            f.write(objstr % ('face', bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]))
        f.write(tail)

# Function to convert VOC format bounding box to YOLO format
def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0  # center x = (xmin + xmax) / 2
    y = (box[2] + box[3]) / 2.0  # center y = (ymin + ymax) / 2
    w = box[1] - box[0]          # width = xmax - xmin
    h = box[3] - box[2]          # height = ymax - ymin
    x = x * dw                   # normalize by image width
    w = w * dw                   # normalize by image width
    y = y * dh                   # normalize by image height
    h = h * dh                   # normalize by image height
    return (x, y, w, h)

def convert_annotation(image_id):
    in_file = open(f'Annotations/{image_id}.xml')
    out_file = open(f'labels/{image_id}.txt', 'w')
    
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert_to_yolo((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    
    in_file.close()
    out_file.close()


# Function to create needed directories
def create_directories():
    directories = ['Annotations', 'ImageSets/Main', 'images', 'labels']
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)


# Function to process WIDER dataset and convert to intermediate VOC format
def process_wider_data(idx, datatype):
    sets_file = open(f'ImageSets/Main/{datatype}.txt', 'a')
    bbx_file = open(f'wider_face_split/wider_face_{datatype}_bbx_gt.txt', 'r')
    
    while True:
        filename = bbx_file.readline().strip('\n')
        if not filename:
            break
            
        # Read image and get dimensions
        img_path = f'WIDER_{datatype}/images/{filename}'
        im = io.imread(img_path)
        head = headstr % (idx, im.shape[1], im.shape[0], im.shape[2])
        
        # Process bounding boxes
        nums = bbx_file.readline().strip('\n')
        bbxes = []
        if nums == '0':
            bbx_info = bbx_file.readline()
            continue
            
        for ind in range(int(nums)):
            bbx_info = bbx_file.readline().strip(' \n').split(' ')
            bbx = [int(bbx_info[i]) for i in range(len(bbx_info))]
            # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
            if bbx[7] == 0:  # Only keep valid faces
                bbxes.append(bbx)
                
        # Skip if no valid faces
        if not bbxes:
            continue
            
        # Write XML annotation and copy image
        writexml(idx, head, bbxes, tailstr)
        shutil.copyfile(img_path, f'images/{idx:06d}.jpg')
        sets_file.write(f'{idx:06d}\n')
        idx += 1
        
    sets_file.close()
    bbx_file.close()

def main():
    # Step 1: Create necessary directories
    create_directories()
    
    # Step 2: Process WIDER Face dataset and convert to VOC format
    idx = 1
    idx = process_wider_data(idx, 'train')
    idx = process_wider_data(idx, 'val')
    
    # Step 3: Convert VOC format to YOLO format
    wd = getcwd()
    for image_set in ['train', 'val']:
        image_ids = open(f'ImageSets/Main/{image_set}.txt').read().strip().split()
        list_file = open(f'{image_set}.txt', 'w')
        
        for image_id in image_ids:
            # Write image path to dataset list file
            line = f'{wd}/images/{image_id}.jpg\n'
            list_file.write(line.replace('\\', '/'))
            # Convert annotation to YOLO format
            convert_annotation(image_id)
            
        list_file.close()
    
    print("Conversion completed! Dataset is ready for YOLO training.")

if __name__ == '__main__':
    main()
