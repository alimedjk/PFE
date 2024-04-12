
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QSlider, QHBoxLayout, QFileDialog, QTextEdit

from mpl_toolkits import mplot3d
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from stl import mesh


import os
import cv2
import numpy as np

from Vnet.model_vnet3d import Vnet3dModule
from Vnet.layer import save_images

#from __future__ import print_function, division
from glob import glob

def divide_images_in_folder(folder_path):
    # Create the output folder
    output_folder ="segmentation/Tranche"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of directories in the folder
    directories = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # Loop through each directory
    for directory in directories:
        # Get the directory path
        directory_path = os.path.join(folder_path, directory)

        # Create the subfolder in the output folder
        subfolder = os.path.join(output_folder, directory)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        # Get the image file paths
        src_img_path = os.path.join(directory_path, 'test_src.bmp')
        predict_img_path = os.path.join(directory_path, 'test_predict.bmp')

        # Check if the image files exist
        if os.path.exists(src_img_path) and os.path.exists(predict_img_path):
            # Load the images
            img = cv2.imread(src_img_path)
            mask = cv2.imread(predict_img_path, 0)
            img = cv2.bitwise_and(img, img, mask=mask)

            # Determine the size of each smaller image
            num_rows = 4
            num_cols = 4
            height, width, _ = img.shape
            small_height = height // num_rows
            small_width = width // num_cols

            # Loop through each row and column
            for i in range(num_rows):
                for j in range(num_cols):
                    # Calculate the coordinates of the smaller image
                    x1 = j * small_width
                    y1 = i * small_height
                    x2 = x1 + small_width
                    y2 = y1 + small_height

                    # Extract the smaller image
                    small_img = img[y1:y2, x1:x2]

                    # Save the smaller image in the subfolder
                    output_file = os.path.join(subfolder, f'small_img_{i}_{j}_{directory}.tif')
                    output_file2 = os.path.join(subfolder, f'small_img_{i}_{j}_{directory}.png')
                    cv2.imwrite(output_file, small_img)
                    cv2.imwrite(output_file2, small_img)

    return output_folder

import os
import meshlib.mrmeshpy as mr

def create_3d_images_from_folders(folder_path):
    output_folder = "segmentation/Slices"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Specify the settings for loading the TIFF images
    settings = mr.LoadingTiffSettings()
    
    # Specify the size of the 3D image element
    settings.voxelSize = mr.Vector3f(1, 1, 5)
    
    # Get list of directories in the main folder
    directories = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    # Loop through each directory
    for directory in directories:
        # Get the directory path
        directory_path = os.path.join(folder_path, directory)
        
        # Update the directory path in the settings
        settings.dir = directory_path
        
        # Create an empty voxel object
        volume = mr.loadTiffDir(settings)
        
        # Specify the ISO value to build the surface
        iso = 127.0
        
        # Convert the voxel object to a mesh
        mesh = mr.gridToMesh(volume, iso)
        
        # Save the mesh to the output folder
        output_file = os.path.join(output_folder, f"{directory}.stl")
        mr.saveMesh(mesh, mr.Path(output_file))
    return output_folder


def getRangImageDepth(image):
    """
    :param image:
    :return:rang of image depth
    """
    # start, end = np.where(image)[0][[0, -1]]
    fistflag = True
    startposition = 0
    endposition = 0
    for z in range(image.shape[0]):
        notzeroflag = np.max(image[z])
        if notzeroflag and fistflag:
            startposition = z
            fistflag = False
        if notzeroflag:
            endposition = z
    return startposition, endposition
def resize_image_itk(itkimage, newSpacing, resamplemethod=sitk.sitkNearestNeighbor):
    """
    image resize withe sitk resampleImageFilter
    :param itkimage:
    :param newSpacing:such as [1,1,1]
    :param resamplemethod:
    :return:
    """
    newSpacing = np.array(newSpacing, float)
    originSpcaing = itkimage.GetSpacing()
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    factor = newSpacing / originSpcaing
    newSize = originSize / factor
    newSize = newSize.astype(np.int64)
    resampler.SetReferenceImage(itkimage)
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetSize(newSize.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    if resamplemethod == sitk.sitkNearestNeighbor:
        itkimgResampled = sitk.Threshold(itkimgResampled, 0, 1.0, 255)
    imgResampled = sitk.GetArrayFromImage(itkimgResampled)
    return imgResampled, itkimgResampled
def load_itk(filename):
    """
    load mhd files and normalization 0-255
    :param filename:
    :return:
    """
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    # Reads the image using SimpleITK
    itkimage = rescalFilt.Execute(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))
    return itkimage

def load_itkfilewithtrucation(filename, upper=200, lower=-200):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of liver value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage

def processOriginaltraindata(file_path):
    expandslice = 13
    trainImage = "segmentation/im/"
    trainMask = "segmentation/ma/"
    
    """
    Charge l'image itk, change la valeur de l'espacement z à 1, et enregistre l'image.
    :param file_path: Chemin du fichier mhd
    :return: None
    """
    file_path_mask = os.path.splitext(file_path)[0]
    file_path_mask = file_path_mask+ "_segmentation.mhd"
    print(file_path_mask)
    # Charger l'image itk et tronquer les valeurs avec une borne supérieure et inférieure
    src = load_itkfilewithtrucation(file_path, 600, -1000)
    sub_img_file = os.path.splitext(os.path.basename(file_path))[0]
    seg = load_itkfilewithtrucation(file_path_mask, 600, -1000)
    
    # Changer l'espacement z > 1.0 à 1.0
    segzspace = seg.GetSpacing()[-1]
    if segzspace > 1.0:
        _, seg = resize_image_itk(seg, (seg.GetSpacing()[0], seg.GetSpacing()[1], 1.0),
                                          resamplemethod=sitk.sitkNearestNeighbor)
        _, src = resize_image_itk(src, (src.GetSpacing()[0], src.GetSpacing()[1], 1.0),
                                  resamplemethod=sitk.sitkLinear)
    
    # Obtenir le tableau d'échantillonnage (image)
    segimg = sitk.GetArrayFromImage(seg)
    srcimg = sitk.GetArrayFromImage(src)

    trainimagefile = trainImage + sub_img_file
    trainmaskfile = trainMask + sub_img_file
    
    if not os.path.exists(trainimagefile):
        os.makedirs(trainimagefile)
    if not os.path.exists(trainmaskfile):
        os.makedirs(trainmaskfile)
    seg_liverimage = segimg.copy()
    seg_liverimage[segimg > 0] = 255
    # Obtenir la plage roi du masque, et étendre le nombre de tranches avant et après, et obtenir l'image roi étendue
    startpostion, endpostion = getRangImageDepth(seg_liverimage)
    if startpostion == endpostion:
        return
    imagez = np.shape(seg_liverimage)[0]
    startpostion = startpostion - expandslice
    endpostion = endpostion + expandslice
    if startpostion < 0:
        startpostion = 0
    if endpostion > imagez:
        endpostion = imagez
    srcimg = srcimg[startpostion:endpostion, :, :]
    seg_liverimage = seg_liverimage[startpostion:endpostion, :, :]
    
    # Écrire l'image src
    # Écrire l'image src
    for z in range(seg_liverimage.shape[0]):
        srcimg = np.clip(srcimg, 0, 255).astype('uint8')
        image_path = os.path.join(trainimagefile, str(z) + ".bmp")
        mask_path = os.path.join(trainmaskfile, str(z) + ".bmp")
        cv2.imwrite(image_path, srcimg[z])
        cv2.imwrite(mask_path, seg_liverimage[z])
    return trainImage, trainMask



def subimage_generator(image, mask, patch_block_size, numberxy, numberz):
    """
    generate the sub images and masks with patch_block_size
    :param image:
    :param patch_block_size:
    :param stride:
    :return:
    """
    width = np.shape(image)[1]
    height = np.shape(image)[2]
    imagez = np.shape(image)[0]
    block_width = np.array(patch_block_size)[1]
    block_height = np.array(patch_block_size)[2]
    blockz = np.array(patch_block_size)[0]
    stridewidth = (width - block_width) // numberxy
    strideheight = (height - block_height) // numberxy
    stridez = (imagez - blockz) // numberz
    # step 1:if stridez is bigger 1,return  numberxy * numberxy * numberz samples
    if stridez >= 1 and stridewidth >= 1 and strideheight >= 1:
        step_width = width - (stridewidth * numberxy + block_width)
        step_width = step_width // 2
        step_height = height - (strideheight * numberxy + block_height)
        step_height = step_height // 2
        step_z = imagez - (stridez * numberz + blockz)
        step_z = step_z // 2
        hr_samples_list = []
        hr_mask_samples_list = []
        prev_subimage = None
        for z in range(step_z, numberz * (stridez + 1) + step_z, numberz):
            for x in range(step_width, numberxy * (stridewidth + 1) + step_width, numberxy):
                for y in range(step_height, numberxy * (strideheight + 1) + step_height, numberxy):
                    temp1=(mask[z:z + blockz, x:x + block_width, y:y + block_height]!=0).sum()
                    temp2=blockz*block_width*block_height/20.0
                    if temp1>temp2:
                      if prev_subimage is None or not np.array_equal(image[z:z + blockz, x:x + block_width, y:y + block_height],prev_subimage):
                        hr_samples_list.append(image[z:z + blockz, x:x + block_width, y:y + block_height])
                        hr_mask_samples_list.append(mask[z:z + blockz, x:x + block_width, y:y + block_height])
        hr_samples = np.array(hr_samples_list).reshape((len(hr_samples_list), blockz, block_width, block_height))
        hr_mask_samples = np.array(hr_mask_samples_list).reshape(
            (len(hr_mask_samples_list), blockz, block_width, block_height))
        return hr_samples, hr_mask_samples
    # step 2:other sutitation,return one samples
    else:
        nb_sub_images = 1 * 1 * 1
        hr_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float32)
        hr_mask_samples = np.zeros(shape=(nb_sub_images, blockz, block_width, block_height), dtype=np.float32)
        rangz = min(imagez, blockz)
        rangwidth = min(width, block_width)
        rangheight = min(height, block_height)
        hr_samples[0, 0:rangz, 0:rangwidth, 0:rangheight] = image[0:rangz, 0:rangwidth, 0:rangheight]
        hr_mask_samples[0, 0:rangz, 0:rangwidth, 0:rangheight] = mask[0:rangz, 0:rangwidth, 0:rangheight]
        return hr_samples, hr_mask_samples
def make_patch(image, mask, patch_block_size, numberxy, numberz):
    """
    Crée un certain nombre de patches à partir de l'image et du masque
    :param image: Image d'entrée
    :param mask: Masque associé à l'image
    :param patch_block_size: Taille du bloc de patch
    :param numberxy: Nombre de subdivisions en x et y
    :param numberz: Nombre de subdivisions en z
    :return: Sous-images et sous-masques générés
    """
    image_subsample, mask_subsample = subimage_generator(image=image, mask=mask, patch_block_size=patch_block_size,
                                                         numberxy=numberxy, numberz=numberz)
    return image_subsample, mask_subsample

def gen_image_mask(srcimg, seg_image, index, shape, numberxy, numberz, trainImage):
    # step 2 get subimages (numberxy*numberxy*numberz,64, 128, 128)
    sub_srcimages,sub_liverimages = make_patch(srcimg,seg_image, patch_block_size=shape, numberxy=numberxy, numberz=numberz)
    # step 3 only save subimages (numberxy*numberxy*numberz,64, 128, 128)
    samples, imagez = np.shape(sub_srcimages)[0], np.shape(sub_srcimages)[1]
    for j in range(samples):
        sub_masks = sub_liverimages.astype(np.float32)
        sub_masks = np.clip(sub_masks, 0, 255).astype('uint8')
        if np.max(sub_masks[j, :, :, :]) == 255:
         filepath = trainImage + "\\" + str(index) + "_" + str(j) + "\\"
         if not os.path.exists(filepath):
            os.makedirs(filepath)
         for z in range(imagez):
            image = sub_srcimages[j, z, :, :]
            image = image.astype(np.float32)
            image = np.clip(image, 0, 255).astype('uint8')
            cv2.imwrite(filepath + str(z) + ".bmp", image)


def preparenoduledetectiontraindata(srcpath, maskpath):
    height = 512
    width = 512
    trainImage = "segmentation/img6"
    if not os.path.exists(trainImage):
      os.makedirs(trainImage)
    shape = (16, 96, 96)
    numberxy = 30
    numberz = 20

    listsrc = []
    listmask = []
    index = 0

    src_folders = os.listdir(srcpath)
    mask_folders = os.listdir(maskpath)

    for folder in src_folders:
        src_folder_path = os.path.join(srcpath, folder)
        mask_folder_path = os.path.join(maskpath, folder)

        if os.path.isdir(src_folder_path) and os.path.isdir(mask_folder_path):
            for filename in os.listdir(src_folder_path):
                if filename.endswith(".bmp"):
                    image = cv2.imread(os.path.join(src_folder_path, filename), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.imread(os.path.join(mask_folder_path, filename), cv2.IMREAD_GRAYSCALE)
        
                    listsrc.append(image)
                    listmask.append(mask)
                    index += 1

    imagearray = np.array(listsrc)
    maskarray = np.array(listmask)
    imagearray = np.reshape(imagearray, (index, height, width))
    maskarray = np.reshape(maskarray, (index, height, width))
    
    gen_image_mask(imagearray, maskarray, 0, shape=shape, numberxy=numberxy, numberz=numberz, trainImage=trainImage)
 


def is_gray_image(image):
    """
    Vérifie si l'image est entièrement grise.
    :param image: Image d'entrée
    :return: True si l'image est entièrement grise, False sinon
    """
    unique_pixels = np.unique(image)
    return len(unique_pixels) == 1

def count_gray_images(folder_path):
    """
    Compte le nombre d'images entièrement grises dans un dossier.
    :param folder_path: Chemin du dossier contenant les images
    :return: Nombre d'images entièrement grises
    """
    gray_image_count = 0
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if is_gray_image(image):
            gray_image_count += 1
    return gray_image_count

def predict():
    src_dir = "segmentation/img6/"
    output_dir = "segmentation/Predictions/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    Vnet3d = Vnet3dModule(96, 96, 16, channels=1, costname=("dice coefficient",), inference=True,
                          model_path="log\segmeation\model\Vnet3d.pd-50000")
        
    folders = os.listdir(src_dir)
    
    for folder in folders:
        src_path = os.path.join(src_dir, folder)
        output_path = os.path.join(output_dir, folder)
        
        # Ignorer les dossiers avec un grand nombre d'images entièrement grises ou blanches
        gray_image_threshold = 5
        gray_image_count = count_gray_images(src_path)
        if gray_image_count > gray_image_threshold:
            continue
        
        imges = []
        
        for z in range(16):
            img = cv2.imread(os.path.join(src_path, str(z) + ".bmp"), cv2.IMREAD_GRAYSCALE)
            imges.append(img)

        test_imges = np.array(imges)
        test_imges = np.reshape(test_imges, (16, 96, 96))


        prediction = Vnet3d.prediction(test_imges)
        test_images = np.multiply(test_imges, 1.0 / 255.0)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        save_images(test_images, [4, 4], os.path.join(output_path, "test_src.bmp"))
        save_images(prediction, [4, 4], os.path.join(output_path, "test_predict.bmp"))
        
    return output_dir



def load_itkfilewithtruncation(filename, upper=200, lower=-200):
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    sitktruncatedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktruncatedimage.SetSpacing(spacing)
    sitktruncatedimage.SetOrigin(origin)
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktruncatedimage, sitk.sitkFloat32))
    return itkimage

class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.current_screen = 0  # Écran actuel : 0 = écran principal, 1 = écran 2, 2 = écran 3

        # Écran principal
        self.welcome_label = QLabel("Bienvenue dans l'application !")
        self.welcome_label.setAlignment(Qt.AlignCenter)
        self.start_button = QPushButton("Commencer")
        self.start_button.clicked.connect(self.show_screen2)

        # Écran 2
        self.open_button = QPushButton("Ouvrir un fichier")
        self.open_button.clicked.connect(self.open_file)

        # Écran 3
        self.label = QLabel()
        self.slice_label = QLabel()
        self.slice_slider = QSlider(Qt.Horizontal)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_factor = 1.0
        self.current_slice = 0
        self.total_slices = 0
        self.image_array = None
        self.image_info = None
        self.info_text_edit = QTextEdit()
        self.current_file = ""
        self.current_maskfile = ""
        self.current_slicemask = 0
        self.next_slice_button = QPushButton("Slice suivante")
        self.prev_slice_button = QPushButton("Slice précédente")
        self.zoom_in_button = QPushButton("Zoom avant")
        self.zoom_out_button = QPushButton("Zoom arrière")
        self.predict_button = QPushButton("Prédire le masque")
        self.info_button = QPushButton("Info de l'image")
        self.back_button = QPushButton("Revenir")

        self.next_slice_button.clicked.connect(self.next_slice)
        self.prev_slice_button.clicked.connect(self.previous_slice)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.predict_button.clicked.connect(self.predict_mask)
        self.info_button.clicked.connect(self.show_image_info)
        self.back_button.clicked.connect(self.go_back)

        # Créer les mises en page pour chaque écran
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.welcome_label)
        self.main_layout.addWidget(self.start_button)
        self.screen2_layout = QVBoxLayout()
        self.screen2_layout.addWidget(self.open_button)
        self.screen3_layout = QVBoxLayout()
        self.screen3_layout.addWidget(self.label)
        self.screen3_layout.addWidget(self.next_slice_button)
        self.screen3_layout.addWidget(self.prev_slice_button)
        self.screen3_layout.addWidget(self.zoom_in_button)
        self.screen3_layout.addWidget(self.zoom_out_button)
        self.screen3_layout.addWidget(self.predict_button)
        self.screen3_layout.addWidget(self.info_button)
        self.screen3_layout.addWidget(self.slice_label)
        self.screen3_layout.addWidget(self.slice_slider)
        self.screen3_layout.addWidget(self.info_text_edit)
        self.screen3_layout.addWidget(self.back_button)

        # Créer les widgets principaux pour chaque écran
        self.main_widget = QWidget()
        self.screen2_widget = QWidget()
        self.screen3_widget = QWidget()

        # Appliquer les mises en page aux widgets principaux
        self.main_widget.setLayout(self.main_layout)
        self.screen2_widget.setLayout(self.screen2_layout)
        self.screen3_widget.setLayout(self.screen3_layout)

        # Afficher l'écran principal
        self.setCentralWidget(self.main_widget)

    def show_screen2(self):
        self.current_screen = 1
        self.setWindowTitle("Écran 2")
        self.setCentralWidget(self.screen2_widget)

    def show_screen3(self):
        self.current_screen = 2
        self.setWindowTitle("Écran 3")
        self.setCentralWidget(self.screen3_widget)

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Ouvrir un fichier", "", "Fichiers MHD(*.mhd)")
        if filename:
            self.load_image(filename)
            self.show_screen3()

    def load_image(self, filename):
        self.image = load_itkfilewithtruncation(filename)
        self.current_file = filename
        self.current_slice = 0
        self.total_slices = self.image.GetSize()[2]
        self.load_slice()
        self.update_buttons()

    def next_slice(self):
        if self.current_slice < self.image.GetSize()[2] - 1:
            self.current_slice += 1
            self.load_slice()
            self.update_buttons()
            num_slices = self.image.GetSize()[2]
            self.slice_label.setText(f"Image {self.current_slice + 1}/{num_slices}")

    def zoom_in(self):
        self.zoom_factor *= 1.1
        self.load_slice()

    def zoom_out(self):
        self.zoom_factor *= 0.9
        self.load_slice()

    def previous_slice(self):
        if self.current_slice > 0:
            self.current_slice -= 1
            self.load_slice()
            self.update_buttons()
            num_slices = self.image.GetSize()[2]
            self.slice_label.setText(f"Image {self.current_slice + 1}/{num_slices}")

    def load_slice(self):
        slice_image = self.image[:, :, self.current_slice]
        slice_array = sitk.GetArrayFromImage(slice_image)
        slice_image = cv2.resize(slice_array, None, fx=self.zoom_factor, fy=self.zoom_factor, interpolation=cv2.INTER_LINEAR)
        plt.imshow(slice_image, cmap='gray')  # Utilisation de la colormap 'gray' pour le niveau de gris
        plt.axis('off')
        plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
        plt.clf()

        qimage = QImage('temp.png')
        pixmap = QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap)
        self.update()

    def show_image_info(self):
        if self.current_file:
            # Obtenir les informations de l'image
            # Modifier le code suivant en fonction de vos besoins
            info_text = "Fichier Image : {}\n".format(self.current_file)
            info_text += "Slice : {}/{}\n".format(self.current_slice + 1, self.total_slices)
            info_text += "Facteur de zoom : {:.2f}\n".format(self.zoom_factor)

            # Lire le contenu du fichier
            try:
                with open(self.current_file, 'r') as file:
                    file_contents = file.read()
                    info_text += "\nContenu du fichier :\n{}\n".format(file_contents)
            except FileNotFoundError:
                info_text += "\nImpossible de lire le contenu du fichier. Fichier introuvable.\n"
            except IOError as e:
                info_text += "\nUne erreur s'est produite lors de la lecture du fichier : {}\n".format(str(e))

            # Afficher les informations de l'image et le contenu du fichier
            self.info_text_edit.setPlainText(info_text)
            self.info_text_edit.setFocus()

    def predict_mask(self):
        if self.current_file:
            a, b = processOriginaltraindata(self.current_file)
            preparenoduledetectiontraindata(a, b)
            # Effectuer la prédiction du masque
            # Modifier le code suivant en fonction de vos besoins
            input_folder = predict()
            print("Terminé 1")
            output_folder = divide_images_in_folder(input_folder)
            print("Terminé 2")
            sortie = create_3d_images_from_folders(output_folder)
            print("Terminé 3")
            self.display_3d(sortie)

    def display_3d(self, output_file):
        # Charger le fichier STL dans un objet mesh
        output_file = output_file + "/0_0.stl"
        your_mesh = mesh.Mesh.from_file(output_file)

        # Obtenir les sommets des triangles
        vertices = your_mesh.vectors.reshape(-1, 3)
        x, y, z = vertices.T

        # Obtenir les indices des triangles
        tri_idx = np.arange(len(vertices)).reshape(-1, 3)

        # Créer un graphique 3D en utilisant le toolkit mplot3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x, y, z, triangles=tri_idx, cmap='viridis', edgecolor='none')

        # Enregistrer le graphique dans un fichier image
        plt.savefig('temp2.png', bbox_inches='tight', pad_inches=0)
        plt.clf()

        # Charger le fichier image dans un QPixmap
        pixmap = QPixmap('temp2.png')

        # Mettre à l'échelle le QPixmap pour l'adapter à l'étiquette
        pixmap = pixmap.scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio)

        # Définir le QPixmap dans l'étiquette
        self.label.setPixmap(pixmap)
        self.update()

    def update_buttons(self):
        num_slices = self.image.GetSize()[2]
        self.slice_slider.setRange(0, num_slices - 1)
        self.slice_slider.setValue(self.current_slice)
        self.slice_label.setText(f"Image {self.current_slice + 1}/{num_slices}")

        if self.current_slice == 0:
            self.prev_slice_button.setEnabled(False)
        else:
            self.prev_slice_button.setEnabled(True)

        if self.current_slice == num_slices - 1:
            self.next_slice_button.setEnabled(False)
        else:
            self.next_slice_button.setEnabled(True)

    def go_back(self):
       if self.current_screen == 1:  # Vérifier si l'écran actuel est déjà l'écran 2
          return  # Ne rien faire si nous sommes déjà à l'écran 2

       self.current_screen = 1
       self.setWindowTitle("Écran 2")
       self.open_file()

       # Vérifier si la fenêtre principale a déjà été fermée
       if self.centralWidget() is None:
          self.setCentralWidget(self.screen2_widget)
          self.open_file()

    def changeEvent(self, event):
        if event.type() == 99:  # QEvent::WindowStateChange
            self.setStyleSheet("background-color: #0072B2;")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

        if self.current_screen == 2:
            if event.key() == Qt.Key_Right:
                self.next_slice()

            if event.key() == Qt.Key_Left:
                self.previous_slice()

            if event.key() == Qt.Key_Plus:
                self.zoom_in()

            if event.key() == Qt.Key_Minus:
                self.zoom_out()

    def closeEvent(self, event):
        # Supprimer les fichiers temporaires
        if os.path.exists('temp.png'):
            os.remove('temp.png')
        if os.path.exists('temp2.png'):
            os.remove('temp2.png')
        os.remove("segmentation")
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())

import sys
import os
from PyInstaller import __main__ as pyi_main

if getattr(sys, 'frozen', False):
    # Exécution en tant qu'exécutable
    current_dir = sys._MEIPASS
else:
    # Exécution en tant que script Python
    current_dir = os.path.dirname(os.path.abspath(__file__))

# Chemin du dossier Vnet
vnet_dir = os.path.join(current_dir, 'Vnet')

# Appel de PyInstaller pour créer l'exécutable
pyi_main.run([
    '--name=mon_executable',
    '--onefile',
    f'--add-data="{vnet_dir};Vnet"'
    'main.py'
])