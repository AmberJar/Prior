import albumentations as A


def transform_aug(height, width, mean, std, p=0.5):
    """Transform each data of the training dataset.

    Args:
      height: the height of the image and mask.
      width: the width of the image and mask.
      p: probability of random data augmentaion.

    Returns:
      a list containing data augmentaion methods.
    """
    transform = A.Compose(
        [
            A.OneOf(
                [
                    A.ColorJitter(p=p),
                    A.Emboss(p=p),
                    A.HueSaturationValue(p=p),
                    A.ChannelShuffle(p=p),
                ]
            ),

            # A.OneOf(
            #     [
            #         A.ChannelShuffle(p=p),
            #         A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.1),
            #         A.PixelDropout(p=p),
            #     ]
            # ),

            # A.OneOf(
            #     [
            #         A.HorizontalFlip(p=p),
            #         A.Flip(p=p),
            #         A.ShiftScaleRotate(p=p)
            #     ]
            # ),

            A.OneOf(
                [
                    A.RandomScale(scale_limit=0.5, p=p),
                    A.RandomResizedCrop(height=height // 2, width=width // 2, p=0.5),
                    A.CenterCrop(height=height // 2, width=width // 2, p=0.5),
                ]
            ),

            A.Resize(height=height, width=width),
            A.Normalize(mean=mean, std=std)
        ]
    )
    return transform


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import cv2
    from matplotlib import pyplot as plt


    def visualize(image, mask, original_image=None, original_mask=None):
        fontsize = 8
        if original_image is None and original_mask is None:
            fg, ax = plt.subplots(2, 1, figsize=(8, 8))
            ax[0].axis('off')
            ax[0].imshow(image)
            ax[0].set_title('image', fontsize=fontsize)
            ax[1].axis('off')
            ax[1].imshow(mask)
            ax[1].set_title('mask', fontsize=fontsize)
        else:
            fg, ax = plt.subplots(2, 2, figsize=(8, 8))
            ax[0, 0].axis('off')
            ax[0, 0].imshow(original_image)
            ax[0, 0].set_title('Original Image', fontsize=fontsize)
            ax[0, 1].axis('off')
            ax[0, 1].imshow(original_mask)
            ax[0, 1].set_title('Original Mask', fontsize=fontsize)
            ax[1, 0].axis('off')
            ax[1, 0].imshow(image)
            ax[1, 0].set_title('Transformed Image', fontsize=fontsize)
            ax[1, 1].axis('off')
            ax[1, 1].imshow(mask)
            ax[1, 1].set_title('Transformed Mask', fontsize=fontsize)
        plt.show()


    def img_msk_cv2(img_pth, msk_pth):
        """Using opencv lib read image and mask.

        Args:
          img_pth: path of the image.
          msk_pth: path of the mask.

        Returns:
          a list containing image and mask data, image convert BRG to rgb.
        """
        return [cv2.imread(img_pth, -1)[..., ::-1],
                cv2.imread(msk_pth, -1)]


    def img_msk_pil(img_pth, msk_pth):
        """Using PIL lib read image and mask.

        Args:
          img_pth: path of the image.
          msk_pth: path of the mask.

        Returns:
          a list containing image and mask data.
        """
        return [np.asarray(Image.open(img_pth)),
                np.asarray(Image.open(msk_pth))]


    # basic information of images
    img_pth = r"C:\Users\xtao\Desktop\img_msk_pair\10003.tif"
    msk_pth = r"C:\Users\xtao\Desktop\img_msk_pair\10003.png"
    tgt_h, tgt_w = 256, 256

    # do with cv2 as images reader
    img_ori, msk_ori = img_msk_cv2(img_pth, msk_pth)
    trans = transform_aug(tgt_h, tgt_w)(image=img_ori, mask=msk_ori)
    img_aug = trans['image']
    msk_aug = trans['mask']
    print(img_aug.shape)
    print(msk_aug.shape)
    visualize(img_aug, msk_aug, img_ori, msk_ori)

    # do with pil as images reader - Here, I in favor of PIL.
    img_ori, msk_ori = img_msk_pil(img_pth, msk_pth)
    trans = transform_aug(tgt_h, tgt_w)(image=img_ori, mask=msk_ori)
    img_aug = trans['image']
    msk_aug = trans['mask']
    print(img_aug.shape)
    print(msk_aug.shape)
    visualize(img_aug, msk_aug, img_ori, msk_ori)
