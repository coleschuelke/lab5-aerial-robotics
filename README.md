# Lab 5

## Install
### Clone 
```bash
cd ~/Workspace
git clone https://gitlab.com/todd.humphreys/lab5-aerial-robotics.git
cd lab5-aerial-robotics
git submodule update --init --recursive
```

### Build
```bash
cd ~/Workspace/lab5-aerial-robotics
mkdir build 
cd build
cmake ..
make -j
```

### Download balloon images and metadata
```bash
cd ~/Workspace/lab5-aerial-robotics
cd images
wget http://radionavlab.ae.utexas.edu/datastore/aerialRobotics/easy-ones.tar .
tar xf easy-ones.tar
```

## Execute
The initial version of the `locateBalloon` executable (with `main.cc` copied
from `main_post-lecture.cc`), which after the build process is located in the
`exe` directory, requires one input: the path to an image to display.  Run the
executable as follows:
```bash
cd ~/Workspace/lab5-aerial-robotics/exe
./locateBalloon ../images/easy-ones/frame00090.jpg
```

The full version of the `locateBalloon` executable (with `main.cc` copied from
`main_full.cc`), which after the build process is located in the `exe`
directory, requires one input: the path to a directory containing a set of
images and a metadata file named `metadata.log`.  Run the executable as
follows:
```bash
cd ~/Workspace/lab5-aerial-robotics/exe
./locateBalloon -i ../images/easy-ones
```

If you would like to enable the dubugging mode, in which each image is
displayed with contours, along with text information on the aspect ratio,
size, and location of the enclosing rectangle and circle, then tack on a '-d'
at the end of the execution command:

```bash
./locateBalloon -i ../images/easy-ones -d
```

The `opencv_demo` executable (with `opencv-lecture/main.cc` copied from
`opencv-lecture/main_post-lecture.cc`), which after the build process is located
in the `exe` directory, can be run as follows:
```bash
cd opencv-lecture
../exe/opencv_demo
```

## Resources
- [OpenCV 4.2.0 Documentation](https://docs.opencv.org/4.2.0/)
- [Eigen Documentation](http://eigen.tuxfamily.org/dox/)
