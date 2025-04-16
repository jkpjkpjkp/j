import os
import numpy as np
from PIL import Image

def render_parsed(parsed, masks, output_folder):
    """
    Renders each merged mask from the `parsed` structure into a folder of masked images.
    
    Args:
        parsed: The tree structure from the `recurse` function (list for leaves, tuple for internal nodes).
        masks: NumPy array of shape (num_masks, height, width) containing individual masks.
        output_folder: String path to the folder where images will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Use a mutable list to maintain a counter across recursive calls
    counter = [0]
    
    # Define a helper function to compute the merged mask for a subtree
    def merged_mask(subtree):
        """
        Computes the merged mask by OR-ing all individual masks in the subtree.
        
        Args:
            subtree: List of mask indices.
        
        Returns:
            NumPy array (height, width) with values 0 and 255.
        """
        # Initialize an empty mask with the same height and width as the individual masks
        mask = np.zeros(masks.shape[1:], dtype=np.uint8)
        # Combine all masks in the subtree using bitwise OR
        for i in subtree:
            mask |= masks[i]
        return mask
    
    # Define a recursive function to traverse the parsed tree
    def traverse(node):
        """
        Traverses the parsed tree and saves the merged mask for each subtree.
        
        Args:
            node: Current node in the tree (list for leaves, tuple for internal nodes).
        """
        # Extract the subtree list of indices
        if isinstance(node, list):
            # Leaf node: the node itself is the subtree (e.g., [i])
            subtree = node
        else:
            # Internal node: first element of the tuple is the subtree
            subtree = node[0]
        
        # Compute the merged mask for this subtree
        mask = merged_mask(subtree)
        
        # Convert the mask to a PIL Image (mode 'L' for grayscale, expecting 0 and 255)
        img = Image.fromarray(mask, mode='L')
        
        # Generate a unique filename using the counter
        filename = f"mask_{counter[0]:03d}.png"
        img.save(os.path.join(output_folder, filename))
        
        # Increment the counter
        counter[0] += 1
        
        # If the node is an internal node (tuple), recurse on children
        if isinstance(node, tuple):
            traverse(node[1])  # left_child
            traverse(node[2])  # right_child
    
    # Start the traversal from the root of the parsed tree
    traverse(parsed)