"""
Terrain generation using Perlin noise (Minecraft-style)
"""

import numpy as np
from perlin_noise import PerlinNoise


class TerrainGenerator:
    """Generate heightfields using Perlin noise with reproducible seeding"""

    def __init__(self, seed=42, grid_size=256, grid_spacing=0.1):
        """
        Args:
            seed: Random seed for reproducibility
            grid_size: NxN grid resolution
            grid_spacing: Physical distance between grid points (m)
        """
        self.seed = seed
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        np.random.seed(seed)

    def generate_perlin(
        self,
        scale=1.0,
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
        height_scale=0.2,
        height_offset=0.0,
    ):
        """
        Generate 2D heightfield using layered Perlin noise

        Args:
            scale: Overall scale of noise features (larger = smoother)
            octaves: Number of noise layers
            persistence: Weight of each octave (0-1)
            lacunarity: Frequency multiplier per octave
            height_scale: Maximum height variation
            height_offset: Base height level

        Returns:
            heightfield: (grid_size, grid_size) array of heights
            (x, y) mesh grids in physical coordinates
        """
        # Generate base Perlin noise
        noise = PerlinNoise(octaves=octaves, seed=self.seed)

        # Generate heightfield with octave layering
        heightfield = np.zeros((self.grid_size, self.grid_size))
        amplitude = 1.0
        frequency = 1.0 / scale
        max_amplitude = 0.0

        for octave in range(octaves):
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    # Sample coordinates scaled by frequency
                    x = (i / self.grid_size) * frequency
                    y = (j / self.grid_size) * frequency

                    # Get Perlin value [-1, 1]
                    value = noise([x, y])

                    # Add to heightfield with current amplitude
                    heightfield[i, j] += value * amplitude

            # Update for next octave
            amplitude *= persistence
            frequency *= lacunarity
            max_amplitude += amplitude

        # Normalize and scale
        if max_amplitude > 0:
            heightfield = heightfield / max_amplitude

        heightfield = heightfield * height_scale + height_offset

        return heightfield

    def generate_flat(self, height=0.0):
        """Generate a flat terrain"""
        return np.full((self.grid_size, self.grid_size), height)

    def generate_stairs(self, step_height=0.1, step_width=5):
        """Generate staircase terrain for difficulty testing"""
        heightfield = np.zeros((self.grid_size, self.grid_size))

        for i in range(self.grid_size):
            step_number = i // step_width
            heightfield[i, :] = step_number * step_height

        return heightfield

    def get_mesh_coordinates(self):
        """Get (x, y) physical coordinates for the heightfield"""
        x = np.arange(self.grid_size) * self.grid_spacing
        y = np.arange(self.grid_size) * self.grid_spacing
        return np.meshgrid(x, y)

    def visualize(self, heightfield, title="Terrain"):
        """Plot heightfield for inspection"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 5))

        # 3D surface
        ax1 = fig.add_subplot(121, projection="3d")
        x, y = self.get_mesh_coordinates()
        ax1.plot_surface(x, y, heightfield, cmap="viridis", alpha=0.9)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Height (m)")
        ax1.set_title(f"{title} - 3D View")

        # 2D heatmap
        ax2 = fig.add_subplot(122)
        im = ax2.imshow(
            heightfield, cmap="viridis", origin="lower", extent=[0, self.grid_size * self.grid_spacing, 0, self.grid_size * self.grid_spacing]
        )
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title(f"{title} - Top View")
        plt.colorbar(im, ax=ax2, label="Height (m)")

        plt.tight_layout()
        plt.show()

    def get_height_at_position(self, heightfield, x, y):
        """
        Get interpolated height at any continuous position

        Args:
            heightfield: (N, N) array
            x, y: Physical coordinates (meters)

        Returns:
            Interpolated height value
        """
        # Convert physical coords to grid indices
        grid_x = x / self.grid_spacing
        grid_y = y / self.grid_spacing

        # Clamp to bounds
        grid_x = np.clip(grid_x, 0, self.grid_size - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size - 1)

        # Bilinear interpolation
        x0, x1 = int(grid_x), min(int(grid_x) + 1, self.grid_size - 1)
        y0, y1 = int(grid_y), min(int(grid_y) + 1, self.grid_size - 1)

        fx = grid_x - x0
        fy = grid_y - y0

        h00 = heightfield[y0, x0]
        h01 = heightfield[y1, x0]
        h10 = heightfield[y0, x1]
        h11 = heightfield[y1, x1]

        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx
        h = h0 * (1 - fy) + h1 * fy

        return h


# Example usage
if __name__ == "__main__":
    gen = TerrainGenerator(seed=42, grid_size=128, grid_spacing=0.1)

    # Generate different types
    print("Generating Perlin terrain...")
    perlin = gen.generate_perlin(
        scale=1.0, octaves=4, persistence=0.5, height_scale=0.2
    )

    print("Generating flat terrain...")
    flat = gen.generate_flat(height=0.0)

    print("Generating stairs...")
    stairs = gen.generate_stairs(step_height=0.1, step_width=5)

    print("Visualizing...")
    gen.visualize(perlin, "Perlin Noise Terrain")
    gen.visualize(flat, "Flat Terrain")
    gen.visualize(stairs, "Staircase Terrain")

    # Test height interpolation
    height = gen.get_height_at_position(perlin, 2.5, 3.7)
    print(f"Height at (2.5, 3.7): {height:.3f} m")
