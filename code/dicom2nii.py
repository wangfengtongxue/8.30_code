
import os
import pydicom
import numpy as np
import nrrd
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
import glob
import SimpleITK as sitk


def get_pixels_hu(path):
    slices = [pydicom.dcmread(path+'/'+slice) for slice in os.listdir(path)]       # path需要以\\结尾，或者这里用os.path.join()
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices])    #将所有的切片堆叠，便于处理
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)   
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)     
    
    return np.array(image, dtype=np.int16),len(slices)

def show_seg(nrrd_path, xml_path,length):
    '''
    3D Slicer保存下来的segmengtation大小不是512*512，需要重新整合成512*512*length
    '''
    nrrd_data, nrrd_options = nrrd.read(nrrd_path)
    seg_data = nrrd_data[0]+nrrd_data[1]           # 0和1表示肝脏和病灶的标注，此处相加整合
    roi_space_origin = list(map(float,nrrd_options['space origin']))            # 标注初始坐标点
    roi_space_directions = list(map(float,[nrrd_options['space directions'][1][0],         #标注体素间距
                                          nrrd_options['space directions'][2][1],
                                          nrrd_options['space directions'][3][2]]))
    
    mrmlFile = xml.dom.minidom.parse(xml_path)
    rootdata = mrmlFile.documentElement
    item_list=rootdata.getElementsByTagName('Volume')     #原始CT的信息
    for it in item_list:
        total_spacing = it.getAttribute('spacing')
        total_origin = it.getAttribute('origin')
    total_spacing_ = []
    total_origin_ = []
    # 初始全是string形式，需要转成数字，也可以用lambda
    for i in total_spacing.split(" "):
        total_spacing_.append(float(i))
    for i in total_origin.split(" "):
        total_origin_.append(float(i))
    print(total_spacing_,total_origin_)
    
    mask = np.zeros([length,512,512])
    # mask在xyz的绝对坐标
    x_start = int(abs(total_origin_[0]-roi_space_origin[0])/total_spacing_[0])
    y_start = int(abs(total_origin_[1]-roi_space_origin[1])/total_spacing_[1])   
    z_start = int(abs(total_origin_[2]-roi_space_origin[2])/total_spacing_[2])

    print(seg_data.shape)
    mask[z_start:z_start+seg_data.shape[2],
        y_start:y_start+seg_data.shape[1],
         x_start:x_start+seg_data.shape[0]]=seg_data.transpose(2,1,0)   
    return np.array(mask, dtype=np.uint8), total_spacing_, total_origin_
# path = r'C:\Users\123\Desktop\zhongshan_data\ZS0021153348\1.2.156.112605.14038004734621.180903063457.3.5292.55020\\'
# slices = [pydicom.dcmread(path+slice) for slice in os.listdir(path)]
# slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))


if __name__ == '__main__':
    base_path = '/data/liver_project/zhongshan_data'    # 标注的数据的地址

    for index, patient_name in enumerate(os.listdir(base_path)):
        patient_path = os.path.join(base_path, patient_name)
        print(patient_path)
        nrrd_path = glob.glob(patient_path+'/slicer/Segmentation.seg.nrrd')[0]
        xml_path = glob.glob(patient_path+'/slicer/*.mrml')[0]

        path = os.path.split(glob.glob(patient_path+'/*/*.dcm')[0])    # 这里的'\\'必须加上，具体原因看get_pixels_hu里第一行
        image, length = get_pixels_hu(path[0])
        mask, total_spacing_, total_origin_ = show_seg(nrrd_path, xml_path, length)
        print(mask.shape, image.shape)

        # 设置方向、初始坐标和体素间距，这些show_seg有输出
        new_ct = sitk.GetImageFromArray(mask)
        new_ct.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        new_ct.SetOrigin((total_origin_[0],total_origin_[1],total_spacing_[2]))
        new_ct.SetSpacing((total_spacing_[0],total_spacing_[1],total_spacing_[2]))
        
        new_im = sitk.GetImageFromArray(image)
        new_im.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        new_im.SetOrigin((total_origin_[0],total_origin_[1],total_spacing_[2]))
        new_im.SetSpacing((total_spacing_[0],total_spacing_[1],total_spacing_[2]))

        seg_save_path = '/data/liver_project/train_file1/segementation/'
        im_save_path = '/data/liver_project/train_file1/volume/'

     #   print(im_save_path + 'volume-{}.nii'.format(str(index)))
        sitk.WriteImage(new_ct, seg_save_path + 'segmentation-{}.nii'.format(str(index)))
        sitk.WriteImage(new_im, im_save_path + 'volume-{}.nii'.format(str(index)))


