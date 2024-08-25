import cv2
import numpy as np
import matplotlib.pyplot as plt


def joint_bilt_filter(D,C,w=5,sigma=(3, 0.1)):
    """
    2-D Joint bilateral filtering for grayscale images.

    Args:
        D (numpy.ndarray): Input grayscale image.
        C (numpy.ndarray): Guiding grayscale image.
        w (int): Half-size of the Gaussian bilateral filter window.
        sigma (tuple): Standard deviations for spatial and intensity domains.

    Returns:
        numpy.ndarray: Filtered image.
    """
    if D is None or not isinstance(D, np.ndarray) or D.dtype != np.float64 or D.min() < 0 or D.max() > 1:
        raise ValueError("Input image D must be a double precision matrix of size NxM on the closed interval [0,1].")

    if C is None or not isinstance(C, np.ndarray) or C.dtype != np.float64 or C.min() < 0 or C.max() > 1:
        raise ValueError("Input image C must be a double precision matrix of size NxM on the closed interval [0,1].")

    w = int(w)
    sigma_d, sigma_r = sigma

    final= np.zeros_like(D)

    X, Y = np.meshgrid(np.arange(-w, w + 1), np.arange(-w, w + 1))
    G = np.exp(-(X**2 + Y**2) / (2 * sigma_d**2))

    dim =D.shape
    a = 0

    while a<dim[0]:
        b=0
        while b<dim[1]:
            iMin=max(a-w,0)
            iMax=min(a+w,dim[0]-1)
            jMin =max(b-w,0)
            jMax =min(b+ w,dim[1]-1)
            I=D[iMin:iMax+1,jMin:jMax+1]
            J=C[iMin:iMax+1,jMin:jMax+1]
            H=np.exp(-((J - C[a, b])**2) / (2 * sigma_r**2))
            F = H * G[iMin - a + w:iMax - a + w + 1, jMin - b + w:jMax - b + w + 1]
            final[a, b] = np.sum(F * I) / np.sum(F)
            b += 1
        a += 1

    return final





def joint_bil_2_color(N, F, w=5, sigma=(3, 0.1)):
    B = np.zeros_like(N)
    for channel in range(N.shape[2]):
        B[:, :, channel] = joint_bilt_filter(N[:, :, channel], F[:, :, channel], w, sigma)
    return B








def solution(image_path_a, image_path_b):
    
    
    path1=image_path_a
    path2=image_path_b

    input_image1 = cv2.imread(path1) / 255.0  # Normalize to [0, 1]
    input_image2 = cv2.imread(path2) / 255.0  # Normalize to [0, 1]

    image_N = np.copy(input_image1)
    image_F = np.copy(input_image2)
    # result = joint_bil_2_color(image_N, image_F, w=11, sigma=(3, 0.1))
    sigma1=3
    sigma2=0.2
    w=11
    print("Calculating Base Image....")
    A_base= joint_bil_2_color(image_N, image_N, w=11, sigma=(3, 0.2))   
    print("calculating Image_NR(Noice reduced version of Image by BLF)....")
    A_nr=joint_bil_2_color(image_N, image_F, w=11, sigma=(3, 0.1))
    print("Calculating Flash Image Base.....")

    F_base=joint_bil_2_color(image_F, image_F, w=11, sigma=(3, 0.1))    

    eps = 0.02
    print("Almost Done....!!")
    F_detail=(image_F.astype(np.float64) + eps) / (F_base.astype(np.float64) + eps)    


    # mask creation

    eps = 0.02
    T = -50  # Set your threshold value here
    i1=cv2.imread(path1)
    i2=cv2.imread(path2)

    if len(i1.shape) > 2:
        gA = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gF = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gA = i1.astype(np.float64)
        gF = i2.astype(np.float64)


    diff=gF-gA; 
    mf = np.zeros_like(diff)
    ms = np.zeros_like(diff)  # Initialize shadow mask
    # Initialize shadow mask

    # Detect shadow where difference is less than or equal to threshold T
    mf[diff <= -50] = 1

    ms[gF / np.max(gF) > 0.95] = 1  # Detect specularities

    M = np.zeros_like(i1)  # Mask initialization
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))  # Structuring element

    # m=mf+ms

    # g = cv2.dilate(m, se)

    for i in range(i1.shape[2]):  # Build the flash mask
        m = np.logical_or(mf, ms).astype(np.uint8)  # Merge two masks
        m=m*255
        M[:, :, i] = cv2.dilate(m, se)

    M=M/255
    
    
    out = ((1 - M) * A_nr * F_detail + M * A_base)

    
    return out

#path_to_no_flash_image=('local/image_nf.png')
#path_to_no_flash_image=('local/image_f.png')

#Clear_image=solution(path_to_no_flash_image,path_to_no_flash_image)

# Convert images from BGR to RGB format
#origional_image=cv2.imread(path_to_no_flash_image)
# image1_rgb = cv2.cvtColor(Clear_image, cv2.COLOR_BGR2RGB)
# image2_rgb = cv2.cvtColor(origional_image, cv2.COLOR_BGR2RGB)

# # Create a figure with two subplots
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# # Display the first image with a title
# axs[0].imshow(image1_rgb)
# axs[0].set_title('BL Filtered Image')
# axs[0].axis('off')  # Hide the axis

# # Display the second image with a title
# axs[1].imshow(image2_rgb)
# axs[1].set_title('Origional Image')
# axs[1].axis('off')  # Hide the axis

# # Display the images
# plt.show()

    
