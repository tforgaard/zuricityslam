### For better reconstruction performance:
```
$ source scripts/colmap_startup.sh

$ cd /cluster/project/infk/courses/252-0579-00L/group07/data/pycolmap

$ python3 -m pip install .

```

### Load packages, i.e. ffmpeg (Euler)
```
$ source scripts/colmap_startup.sh
```

The scripts are useful more intensive jobs

### Load eth_proxy
```
$ module load eth_proxy
```

To submit a batch job, do
```
$ bsub < scripts/single_reconstruction.sh
```
See scripts / code for more info about the different parameters