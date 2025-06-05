import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Path to the image
img_path = 'outputs/test_causality_visualizations/direct_consolidated_graph.png'

if os.path.exists(img_path):
    print(f'Loading image from {img_path}')
    img = mpimg.imread(img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Consolidated Multi-Metric Causality Graph')
    plt.savefig('outputs/consolidated_verification.png')
    print(f'Consolidated image verification saved to outputs/consolidated_verification.png')
else:
    print(f'Image not found at {img_path}')
