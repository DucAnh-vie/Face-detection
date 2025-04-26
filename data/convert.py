# -*- coding: utf-8 -*-
import shutil
import os
import xml.etree.ElementTree as ET
from os import getcwd
from skimage import io

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

classes = ["face"]

sets = ['train', 'val']


def writexml(idx, head, bbxes, tail):
    filename = ("Annotations/%06d.xml" % (idx))
    f = open(filename, "w")
    f.write(head)
    for bbx in bbxes:
        f.write(objstr % ('face', bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]))
    f.write(tail)
    f.close()


def clear_dir():
    if shutil.os.path.exists(('Annotations')):
        shutil.rmtree(('Annotations'))
    if shutil.os.path.exists(('ImageSets')):
        shutil.rmtree(('ImageSets'))
    if shutil.os.path.exists(('images')):
        shutil.rmtree(('images'))
    if not os.path.exists('labels/'):
        os.makedirs('labels/')
    shutil.os.mkdir(('Annotations'))
    shutil.os.makedirs(('ImageSets/Main'))
    shutil.os.mkdir(('images'))


def excute_datasets(idx, datatype):
    f = open(('ImageSets/Main/' + datatype + '.txt'), 'a')
    f_bbx = open(('wider_face_split/wider_face_' + datatype + '_bbx_gt.txt'), 'r')
    while True:
        filename = f_bbx.readline().strip('\n')
        if not filename:
            break
        im = io.imread(('WIDER_' + datatype + '/images/' + filename))
        head = headstr % (idx, im.shape[1], im.shape[0], im.shape[2])
        nums = f_bbx.readline().strip('\n')
        bbxes = []
        if nums == '0':
            bbx_info = f_bbx.readline()
            continue
        for ind in range(int(nums)):
            bbx_info = f_bbx.readline().strip(' \n').split(' ')
            bbx = [int(bbx_info[i]) for i in range(len(bbx_info))]
            # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
            if bbx[7] == 0:
                bbxes.append(bbx)
        writexml(idx, head, bbxes, tailstr)
        shutil.copyfile(('WIDER_' + datatype + '/images/' + filename), ('images/%06d.jpg' % (idx)))
        f.write('%06d\n' % (idx))
        idx += 1
    f.close()
    f_bbx.close()
    return idx


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open('Annotations/%s.xml' % (image_id))
    out_file = open('labels/%s.txt' % (image_id), 'w')
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
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def create_yaml_file():
    # Dynamically determine the base path
    base_path = os.getcwd().replace("\\", "/")

    yaml_content = f"""\
# Train/val/test set as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: '{base_path}' # dataset root dir
train: images #train images (relative to 'path')
val: images #val images (relative to 'path')
train_label_dir: labels
val_label_dir: labels
# Classes
names: 
    - face
"""
    # Write the YAML content to a file
    with open('dataset_config.yml', 'w') as yaml_file:
        yaml_file.write(yaml_content)
    print("YAML file 'dataset_config.yml' created successfully.")


if __name__ == '__main__':
    # First part from convert.py
    print("Starting conversion process...")
    clear_dir()
    idx = 1
    idx = excute_datasets(idx, 'train')
    idx = excute_datasets(idx, 'val')
    print('XML generation complete...')

    wd = getcwd()
    for image_set in sets:
        image_ids = open('ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
        list_file = open('%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            line = '%s/images/%s.jpg\n' % (wd, image_id)
            list_file.write(line.replace("\\", '/'))
            convert_annotation(image_id)
        list_file.close()

    create_yaml_file()

    print("All processing completed...")
