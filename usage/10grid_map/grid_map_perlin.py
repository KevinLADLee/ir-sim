"""
Example: Perlin noise based grid map.

Generate a procedural occupancy map with Perlin2dMap, save as PNG, and run
a simulation. Parameter semantics follow GCOPTER map_gen
(https://github.com/ZJU-FAST-Lab/GCOPTER).
"""

import irsim
from irsim.world.map import Perlin2dMap

pmap = (
    Perlin2dMap(
        width=200,
        height=200,
        complexity=0.12,
        fill=0.32,
        fractal=1,
        attenuation=0.5,
        seed=48,
    )
    .generate()
)

# pmap.save_as_image("perlin_cave.png")
print(f"Generated map: {pmap.grid.shape}, obstacle ratio: {(pmap.grid > 50).mean():.1%}")

env = irsim.make("grid_map_perlin.yaml", save_ani=False, full=False)

for _ in range(500):
    env.step()
    env.render()

    if env.done():
        break

env.end(5)
