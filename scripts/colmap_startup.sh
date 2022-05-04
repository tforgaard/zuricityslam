export BASE=/cluster/project/infk/courses/252-0579-00L/group07
export DATA=$BASE/data

# Fix write/read permissions user:rwx, group:rwx, others:r
umask 002

# COLMAP

module load gcc/6.3.0

module load cmake/3.16.5

module load boost eigen glog gflags glew

module load cgal openblas suite-sparse atlas

module load mpfr

module load mesa mesa-glu

module load libxcb

module load cuda/10.0.130

module load metis

module load libx11

# PREPROCESSING

module load ffmpeg

# VARS

export FREEIMAGE_DIR=$DATA/dev/FreeImage/Dist

export CERES_DIR=$DATA/dev/ceres-solver-1.14.0/build

export Qt5_DIR=$DATA/dev/qt5-build/qtbase/lib/cmake/Qt5

export COLMAP_PATH=$DATA/colmap/usr/local/bin
export COLMAP_DIR=$DATA/colmap/usr/local/share/colmap

export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$FREEIMAGE_DIR

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DATA/dev/qt5-build/qtbase/lib

export PATH=$PATH:/$DATA/colmap/usr/local/bin
