import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import SimpleITK as itk
import imageio
from math import ceil

def show_nii(nii_file_path,max_slices_num=8) -> None:
    """ Show the NII file by plotting a few slices of it.

    Args:
        nii_file_path: the absolute path of NII file 
            e.g. D:/microsoft_PBL/dataset/volume_pt1/volume-0.nii
    
    Raises when:
        1. nii_file_path is not an absolute path
        2. the suffix of the file is not ".nii"
    """
    assert(os.path.isabs(nii_file_path))
    _,suffix=os.path.splitext(nii_file_path)
    assert(suffix=='.nii')

    img=nib.load(nii_file_path)
    (_,_,queue)=img.dataobj.shape

    No=1
    for i in range(0,queue,ceil(queue/max_slices_num)):
        img_arr=img.dataobj[:,:,i]
        plt.subplot(5,4,No)
        No+=1
        plt.imshow(img_arr,cmap='gray')
    
    plt.show()

def pre_process(nii_file_path, output_file_path=None, new_spacing=[0.7,0.7,1.0]) -> None:
    """Process the NII file so that it can be accpected by the neural network.
    The processed NII file's name will end with "_done". For instance, originally 
    there is a NII file "abc.nii"; following processing there will be a new file
    "abc_done.nii".
    
    Args:
        nii_file_path: the absolute path of NII file
        output_file_path: the absolute path where the processed NII file will be saved.
            If empty, then the processed NII file will be saved into the same directory
            with the original NII file.
        
    """
    assert(os.path.isabs(nii_file_path))
    prefix,suffix=os.path.splitext(nii_file_path)
    assert(suffix=='.nii')

    if output_file_path==None:
        output_file_path=os.path.join(nii_file_path,'..','{}_jiaren.nii'.format(prefix))
    assert(os.path.isabs(output_file_path))

    img=itk.ReadImage(nii_file_path)
    original_spacing=img.GetSpacing()
    original_size=img.GetSize()

    new_size=(
        int(np.round(original_size[0] * original_spacing[0] / new_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / new_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / new_spacing[2])),
    )

    resample=itk.ResampleImageFilter()
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing)

    if("segmentation" in prefix):
        resample.SetInterpolator(itk.sitkNearestNeighbor)
        resample.SetOutputPixelType(itk.sitkUInt8)
    else:
        resample.SetInterpolator(itk.sitkLinear)
        resample.SetOutputPixelType(itk.sitkFloat32)
    
    new_img=resample.Execute(img)
    itk.WriteImage(new_img,output_file_path)

def nii_to_png(nii_file_path, max_slices_num=20 ,output_dir=None):
    """Transfer NII file to PNG files.

    Args:
        nii_file_path: the absolute path of NII file
        output_dir: saves PNG images
        max_slices_num
    """

    assert(os.path.isabs(nii_file_path))
    prefix,suffix=os.path.splitext(nii_file_path)
    assert(suffix=='.nii')
    assert(os.path.exists(nii_file_path))

    if(output_dir==None):
        output_dir=os.path.join(nii_file_path,'..',prefix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    assert(os.path.isabs(output_dir))
    assert(os.path.exists(output_dir))

    img=nib.load(nii_file_path)
    img_fdata = img.get_fdata()

    # transfer to PNG
    (_,_,queue)=img.shape

    No=0
    for i in range(0,queue,ceil(queue/max_slices_num)):
        No+=1
        slice = img_fdata[:, :, i]  # Choose Z-axis
        imageio.imwrite(os.path.join(output_dir, '{}.png'.format(No)), slice)

if __name__=='__main__':
    ct_fil='D:/microsoft_PBL/dataset/volume_pt1/volume-7.nii'
    nii_to_png(ct_fil)
    print('done')

if __name__=='__main__2':
    ct_fil='D:/microsoft_PBL/dataset/volume_pt1/volume-2.nii'
    show_nii(ct_fil)
    print(os.listdir(ct_fil+'/..'))

    prefix,_=os.path.splitext(ct_fil)

    pre_process(ct_fil)
    show_nii(os.path.join(ct_fil,'..',prefix+'_done.nii'))
