# PR 草稿：Perlin 程序化 2D 地图

GitHub 新建 Pull Request 时，标题与正文可参考以下内容。

---

## 标题 (Title)

```
feat(map): add Perlin noise map generator and usage example
```

---

## 正文 (Body)

```markdown
## Summary

- Add `Perlin2dMap` for procedural 2D occupancy grids using Perlin noise
- Parameters (`complexity`, `fill`, `fractal`, `attenuation`) aligned with GCOPTER-style map generation
- Optional `seed` for reproducibility; pure NumPy, no extra dependencies
- Export as PNG for use with existing `obstacle_map` in YAML
- Register in `irsim.world.map`; add usage example under `usage/10grid_map/`

## Usage

```python
from irsim.world.map import Perlin2dMap

pmap = (
    Perlin2dMap(width=200, height=200, complexity=0.12, fill=0.32, seed=48)
    .generate()
)
pmap.save_as_image("perlin_cave.png")
# Use perlin_cave.png as obstacle_map in YAML
```

See `usage/10grid_map/grid_map_perlin.py` and `grid_map_perlin.yaml`.

## Test plan

- [x] `ruff check` on changed files
- [x] Unit tests in `tests/test_perlin_map.py`
- [ ] Run `usage/10grid_map/grid_map_perlin.py` locally (optional)
```
