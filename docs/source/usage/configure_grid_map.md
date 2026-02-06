Configure grid map environment
==============================

The grid map environment is a 2D grid-based environment that can be used to simulate various scenarios. It can be simply configured by specifying path of image file in the YAML configuration file. 


## Grid Map Configuration Parameters

The python script and YAML configuration file for the grid map environment are shown below:

::::{tab-set}

:::{tab-item} Python Script

```python

import irsim

env = irsim.make()

for i in range(1000):

    env.step()
    env.render(0.05)

    if env.done():
        break

env.end()
```

:::

:::{tab-item} YAML Configuration

```yaml
world:
  height: 50  
  width: 50  
  obstacle_map: 'cave.png'
  mdownsample: 2

robot:
  - kinematics: {name: 'acker'} 
    shape: {name: 'rectangle', length: 4.6, width: 1.6, wheelbase: 3}
    state: [5, 5, 0, 0]
    goal: [40, 40, 0]
    vel_max: [4, 1]
    plot:
      show_trail: True
      traj_color: 'g'
      show_trajectory: True
      show_goal: False

    sensors: 
      - name: 'lidar2d'
        range_min: 0
        range_max: 20
        angle_range: 3.14
        number: 100
        alpha: 0.4


obstacle:
  - number: 10
    distribution: {name: 'manual'}
    shape:
      - {name: 'polygon', random_shape: true, center_range: [5, 10, 40, 30], avg_radius_range: [0.5, 2]} 

```

:::

:::{tab-item} Demonstration
:selected:

```{image} gif/grid_map.gif
:alt: Select Parameters
:width: 400px
:align: center
```
:::
::::

### Important Parameters Explained

To configure the grid map environment, use the single key **obstacle_map** in the `world` section:

1. **Default — image file**: Set `obstacle_map` to a path string (e.g. `'cave.png'`). This uses the image generator internally; existing YAMLs need no change. The `mdownsample` parameter downsamples the grid for acceleration.
2. **Procedural / other generators**: Set `obstacle_map` to an object with `name` and `resolution` (meters per cell). Grid size is computed from the world `width` and `height` (e.g. world 20×20 m with `resolution: 0.1` → 200×200 cells).

   ```yaml
   world:
     height: 20
     width: 20
     mdownsample: 1
     obstacle_map:
       name: perlin
       resolution: 0.1   # 20 m / 0.1 = 200 cells per axis
       complexity: 0.12
       fill: 0.32
       fractal: 1
       attenuation: 0.5
       seed: 48   # optional; omit for random map each run
   ```

   See ``usage/10grid_map/grid_map_perlin.yaml`` and ``grid_map_perlin.py`` for a full example.

   To add a new generator (e.g. maze), implement a subclass of ``irsim.world.map.GridMapGenerator`` with ``_build_grid()``, set class attributes ``name`` and ``yaml_param_names``, and import it in ``irsim.world.map`` so it registers; then use ``obstacle_map: { name: your_name, ... }`` in YAML.

The image of `cave.png` (when using an image) should be placed in the same directory as the python script, and is shown below:

```{image} ../cave.png
:alt: Select Parameters
:width: 400px
:align: center
```

In the simulation, this png figure will be rasterized into a grid map. Black pixels represent obstacles, and white pixels represent free space. 

:::{tip}
You can use custom png images to create different grid map environments. The absolute or relative paths can be used to specify the image file in other directories.
:::