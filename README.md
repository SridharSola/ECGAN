

# ECGAN
This is the repository for our work 'Unmaksing Your Expression: Expression-Conditioned GAN for Masked Face Inpainting' presented at the 5th Workshop on Affective Behaviour Analysis in the Wild part of CVPR2023 Workshops. The paper is available [here](https://openaccess.thecvf.com/content/CVPR2023W/ABAW/papers/Sola_Unmasking_Your_Expression_Expression-Conditioned_GAN_for_Masked_Face_Inpainting_CVPRW_2023_paper.pdf). Alternatively, you can watch [this](https://drive.google.com/file/d/1S_q8ZUrGz617OED72Pveeem67_-BndtD/view?usp=sharing) short video presentation. To view the code in a notebook, check out this [Colab file](https://colab.research.google.com/drive/1zghk01Dy1vlGpBIGRIfKDctpMq9NLio8#scrollTo=BcO7vrKsE54G).

The starter code for our conditioned GAN (vanilla UNet, loss, train, and test functions) are taken from [this project](https://github.com/daviddirethucus/Face-Mask_Inpainting.git).

**TL;DR** <br/>
ECGAN takes in a masked image, its mask binary segmentation, and an expression class, and returns an unmasked image with the expression as shown:

![](example.gif)


