import matplotlib.pyplot as plt

def unnormalize(tensor):
    return tensor * 0.5 + 0.5

def show_before_after(raw_image, transformed_image):
    plt.figure(figsize=(20, 4))

    # raw image
    plt.subplot(1, 2, 1)
    plt.title("Before Transforms")
    plt.imshow(raw_image, cmap="gray")
    plt.axis("off")

    # transformed image
    plt.subplot(1, 2, 2)
    plt.title("After Transforms")
    image = unnormalize(transformed_image).permute(1, 2, 0).squeeze() # (C, H, W) -> (H, W, C) -> (H, W)
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.show()
