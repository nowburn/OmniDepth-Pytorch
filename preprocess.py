import sys
import numpy as np
import cv2
from PIL import Image
import os
import re
import shutil


# imgs_path = './data/test/'
# output_path = './data/panoimgs/result.jpg'
# img_list = [imgs_path + str(num) + '.jpeg' for num in range(1, 3)]
# depth_json_path = './data/test/U4qAucYgiXcf4P3kDmXxiQ.json'


def stitch_pano_overlap(img_list, output_path):
    imgs = []
    for img_name in img_list:
        img = cv2.imread(img_name)
        if img is None:
            print("can't read image " + img_name)
            sys.exit(-1)
        imgs.append(img)
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(imgs)

    if status != cv2.Stitcher_OK:
        print("Can't stitch images, error code = %d" % status)
        sys.exit(-1)
    cv2.imwrite(output_path, pano)
    print("stitching completed successfully. %s saved!" % output_path)


def stitch_pano(img_list, output_path):
    imgs = []
    imgs_size = []
    for item in img_list:
        img = Image.open(item)
        imgs.append(img)
        imgs_size.append(img.size)
    new_size = np.sum(imgs_size, axis=0)
    joint = Image.new('RGB', (new_size[0], imgs_size[0][1]))
    loc = []
    x = 0
    loc.append((x, 0))
    for item in imgs_size:
        x += list(item)[0]
        loc.append((x, 0))
    for i, img in enumerate(imgs):
        joint.paste(img, loc[i])
    joint.save(output_path)


def pano_resize(imgs_path):
    img_cnt = 0
    for img_name in os.listdir(imgs_path):
        if not re.match(r'.+d\.jpeg', img_name):
            img_cnt += 1
            img_path = imgs_path + img_name
            img = Image.open(img_path)
            (x, y) = img.size
            x_new = 512
            y_new = int(y * x_new / x)
            out = img.resize((x_new, y_new), Image.ANTIALIAS)
            out.save(img_path)

    print("resized imgs count:", img_cnt)


def delete_nopair_imgs(imgs_path):
    img_list = []
    del_cnt = 0
    for img_name in os.listdir(imgs_path):
        if re.match(r'.+d\.jpeg', img_name):
            prefix = img_name[:-7]
        else:
            prefix = img_name[:-5]

        if prefix in img_list:
            img_list.remove(prefix)
        else:
            img_list.append(prefix)
    num_wrong_imgs = len(img_list)
    print('find no pair or repeat imgs num: ', num_wrong_imgs)
    for prefix in img_list:
        rgb_img = imgs_path + prefix + ".jpeg"
        depth_img = imgs_path + prefix + "_d.jpeg"
        if os.path.exists(rgb_img):
            os.remove(rgb_img)
            del_cnt += 1
        if os.path.exists(depth_img):
            os.remove(depth_img)
            del_cnt += 1
    print('delete no pair or repeat imgs num: ', del_cnt)


def copy_n_imgs(num, origin, to):
    cnt = 0
    for img_name in os.listdir(origin):
        if not re.match(r'.+d\.jpeg', img_name):
            img = Image.open(origin + img_name)
            img.save(to + img_name)
            cnt += 1
        if cnt >= num:
            break
    print('copy {} imgs from {} to {}'.format(cnt, origin, to))


def copy_n_imgpairs(num, origin, to):
    cnt = 0
    img_list = []
    for img_name in os.listdir(origin):
        if re.match(r'.+d\.jpeg', img_name):
            prefix = img_name[:-7]
        else:
            prefix = img_name[:-5]

        if prefix not in img_list:
            img_list.append(prefix)
        cnt += 1
        if cnt >= num:
            break
    for prefix in img_list:
        shutil.move(origin + prefix + '.jpeg', to + prefix + '.jpeg')
        shutil.move(origin + prefix + '_d.jpeg', to + prefix + '_d.jpeg')
    print('copy {} img pairs'.format(cnt))


def filter__nopair_imgs(origin, to):
    img_list = []
    for img_name in os.listdir(origin):
        if re.match(r'.+d\.jpeg', img_name):
            prefix = img_name[:-7]
        else:
            prefix = img_name[:-5]

        if prefix in img_list:
            img_list.remove(prefix)
        else:
            img_list.append(prefix)

    for prefix in img_list:
        shutil.move(origin + prefix + '_d.jpeg', to + prefix + '_d.jpeg')
    print('copy {} img pairs'.format(len(img_list)))


def move_files(origin, to):
    cnt = 0
    for file in os.listdir(origin):
        if re.match(r'.+d\.jpeg', file):
            shutil.move(origin + file, to + file)
            cnt += 1
    print('total move files num: %d' % cnt)


if __name__ == '__main__':
    imgs_path = '/home/nowburn/python_projects/cv/OmniDepth/show/'
    origin = '/home/nowburn/python_projects/cv/OmniDepth/data/training/'
    to = '/home/nowburn/python_projects/cv/OmniDepth/data/tmp/'
    # copy_n_imgs(100, imgs_path, to)
    # copy_n_imgs(2000, origin, to)
    # copy_n_imgpairs(674, origin, to)
    #filter__nopair_imgs(origin, to)
    pano_resize(imgs_path)


