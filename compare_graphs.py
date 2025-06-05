import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Paths to the images
original_path = 'outputs/test_causality_visualizations/original_granger_graph.png'
improved_path = 'outputs/test_causality_visualizations/direct_improved_granger_graph.png'

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Load and display original graph
if os.path.exists(original_path):
    print(f'Loading original image from {original_path}')
    img1 = mpimg.imread(original_path)
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Original Causality Graph')
else:
    print(f'Original image not found at {original_path}')

# Load and display improved graph
if os.path.exists(improved_path):
    print(f'Loading improved image from {improved_path}')
    img2 = mpimg.imread(improved_path)
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Improved Causality Graph (Nodes Over Edges)')
else:
    print(f'Improved image not found at {improved_path}')

plt.tight_layout()
plt.savefig('outputs/comparison_verification.png')
print(f'Comparison image saved to outputs/comparison_verification.png')
