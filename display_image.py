import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Path to the image
img_path = 'outputs/test_causality_visualizations/direct_improved_granger_graph.png'

if os.path.exists(img_path):
    print(f'Loading image from {img_path}')
    img = mpimg.imread(img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Improved Causality Graph')
    plt.savefig('outputs/image_verification.png')
    print(f'Image verification saved to outputs/image_verification.png')
else:
    print(f'Image not found at {img_path}')
