

# ECGAN
This is the repository for our work 'Unmaksing Your Expression: Expression-Conditioned GAN for Masked Face Inpainting' presented at the 5th Workshop on Affective Behaviour Analysis in the Wild part of CVPR2023 Workshops. The paper is available [here](https://openaccess.thecvf.com/content/CVPR2023W/ABAW/papers/Sola_Unmasking_Your_Expression_Expression-Conditioned_GAN_for_Masked_Face_Inpainting_CVPRW_2023_paper.pdf). Alternatively, you can watch [this](https://drive.google.com/file/d/1S_q8ZUrGz617OED72Pveeem67_-BndtD/view?usp=sharing) short video presentation. To view the code in a notebook, check out this [Colab file](https://colab.research.google.com/drive/1zghk01Dy1vlGpBIGRIfKDctpMq9NLio8#scrollTo=BcO7vrKsE54G).

**Abstract.** As face masks continue to be a part of our daily lives,
the challenge of reconstructing occluded faces remains relevant. While several approaches have been proposed for removing masks from neutral facial images, few have explored
the use of facial expressions as a dominant feature for reconstruction of expressive faces. To address this gap, we
propose an expression-conditioned GAN (ECGAN) for reconstructing masked faces with a specified expression. Our
approach leverages both the binary segmentation map of the
mask and an expression label to generate high-quality images. To train our ECGAN in a supervised manner, we synthesize masked images using the RAFDB dataset to create
non-masked-masked pairs of images for training. We evaluate of our approach on the RAFDB test set, demonstrating
its effectiveness in generating realistic images that convincingly belong to the given expression class. This is further
highlighted by comparing it to a baseline model and a stateof-the-art approach without expression-input. 

The starter code for our conditioned GAN (vanilla UNet, loss, train, and test functions) are taken from [this project](https://github.com/daviddirethucus/Face-Mask_Inpainting.git).

**TL;DR** <br/>
ECGAN takes in a masked image, its mask binary segmentation, and an expression class, and returns an unmasked image with the expression as shown:

![](example.gif)


