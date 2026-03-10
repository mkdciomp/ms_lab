import mujoco

import ms_lab.terrains as terrain_gen
from ms_lab.terrains.terrain_generator import TerrainGeneratorCfg
from ms_lab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.4),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.05,
      step_height_range=(0.0, 0.1),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
     proportion=0.05,
     step_height_range=(0.00, 0.1),
     step_width=0.3,
     platform_width=3.0,
      border_width=1.0,
    ),
# "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#        proportion=0.1,
#        slope_range=(0.0, 1.0),
#        platform_width=2.0,
#        border_width=0.25,
#      ),
#     "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
#        proportion=0.05,
#        slope_range=(0.0, 1.0),
#        platform_width=2.0,
#        border_width=0.25,
#        inverted=True,
#      ),
#     "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#        proportion=0.02,
#        noise_range=(0.02, 0.10),
#        noise_step=0.05,
#        border_width=0.25,
#      ),
#     "wave_terrain": terrain_gen.HfWaveTerrainCfg(
#        proportion=0.05,
#        amplitude_range=(0.0, 0.2),
#        num_waves=4.0 ,
#        border_width=0.25,
#      ),
    # NOTE: Heightfield terrains are currently disabled due to compilation issues
    # in mujoco-warp.
},
  add_lights=False,
)
"""

    
  
  
"""


if __name__ == "__main__":
  import mujoco.viewer

  terrain_cfg = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
  )
  terrain = TerrainImporter(terrain_cfg, device="cuda:0")
  mujoco.viewer.launch(terrain.spec.compile())
