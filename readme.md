LIDAR Data from IGN of the zone near le Col de la Bernatoire, converted to STL with python and then processed in blender to have desired thickness/scale.

![](./zone.png)

Thanks IGN for the LiDAR HD project !
https://diffusion-lidarhd.ign.fr/

Command used to generate the STLs :

```
./laz_to_stl.py LHD_FXX_0446_6188_PTS_C_LAMB93_IGN69.copc.laz --max-points 2000000 --voxel-size 0.25 --depth 7
```
