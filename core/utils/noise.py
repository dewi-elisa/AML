import numpy as np


def random_noise(frames, input_length, corruption_ratio=0.1):
    """
    Randomly zeroes out a percentage of pixels in each frame.

    frames: (N, H, W) numpy array
    input_length: int denoting the amount of frames that need to be corrupted for the input
    corruption_ratio: float in [0, 1]
    """
    corrupted = frames.copy()
    N, H, W, C = corrupted.shape
    num_pixels = H * W
    num_to_corrupt = int(num_pixels * corruption_ratio)

    for i in range(input_length):
        indices = np.random.choice(num_pixels, num_to_corrupt, replace=False)
        y_coords, x_coords = np.unravel_index(indices, (H, W))
        corrupted[i, :, y_coords, x_coords] = 0
    return corrupted

def rows_noise(frames, input_length, corruption_ratio=0.1):
    """
    Randomly zeroes out a percentage of rows in each frame.

    frames: (N, H, W) numpy array
    corruption_ratio: float in [0, 1] - percentage of rows to corrupt
    """
    corrupted = frames.copy()
    N, H, W, C = corrupted.shape
    num_rows_to_corrupt = int(H * corruption_ratio)

    for i in range(input_length):
        # Randomly select which rows to corrupt
        row_indices = np.random.choice(H, num_rows_to_corrupt, replace=False)
        # Zero out the selected rows
        corrupted[i, row_indices, :, :] = 0

    return corrupted

def blocks_noise(frames, input_length, corruption_ratio=0.1, block_size=10):
    """
    Randomly zeroes out blocks of pixels in each frame.

    frames: (N, H, W) numpy array
    corruption_ratio: float in [0, 1] - fraction of total area to corrupt
    block_size: int - size of square blocks to corrupt (default 2 for 2x2 blocks)
    """
    corrupted = frames.copy()
    N, C, H, W = corrupted.shape

    # Calculate how many blocks we can fit in each dimension
    blocks_per_row = H // block_size
    blocks_per_col = W // block_size
    total_blocks = blocks_per_row * blocks_per_col

    # Calculate how many blocks to corrupt based on the corruption ratio
    # Each block covers block_size^2 pixels
    total_pixels = H * W
    pixels_to_corrupt = int(total_pixels * corruption_ratio)
    blocks_to_corrupt = max(1, pixels_to_corrupt // (block_size * block_size))

    # Make sure we don't try to corrupt more blocks than available
    blocks_to_corrupt = min(blocks_to_corrupt, total_blocks)

    for i in range(input_length):
        # Generate random block positions
        block_indices = np.random.choice(total_blocks, blocks_to_corrupt, replace=False)

        for block_idx in block_indices:
            # Convert block index to row, col coordinates
            block_row = (block_idx // blocks_per_col) * block_size
            block_col = (block_idx % blocks_per_col) * block_size

            # Zero out the block
            corrupted[i, :, block_row:block_row+block_size, block_col:block_col+block_size] = 0

    return corrupted