from torchvision.utils import save_image
import numpy as np
import cv2

def show_tensor_images(i, image_tensor, num_images=3, size=(3,224,224)):
    #image_tensor = transforms.ToTensor(image_tensor)
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    print(image_unflat.shape)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    #save_image(image_unflat,'/content/drive/MyDrive/Face_Inpainting_1/Binary/' + str(i) + '.jpg')



def improve(i):
  img = cv2.imread('/content/drive/MyDrive/Face_Inpainting_1/Binary/' + str(i) + '.jpg')
  kernel = np.ones((4,4),np.uint8)
  ret, imgg = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
  opening = cv2.morphologyEx(imgg, cv2.MORPH_OPEN, kernel)
  cv2_imshow(opening) #!pip install cv2_imshow
