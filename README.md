# ldbm-image-background-remover

Remove image background automatically with LDBM algorithmn.

*Read this in other languages: [English](README.md), [简体中文](README.zh-cn.md).*

## Demo

[![LDBM Matting](https://github.com/whitelok/ldbm-image-background-remover/blob/master/resources/ldbm.png)](https://github.com/whitelok/ldbm-image-background-remover)

 - Run LDBMImageBackgroundRemover:

```bash
LDBMImageBackgroundRemover /path/of/image /path/of/image_tag
```

## Download Releases

### Version 1.0.0
- [Mac](https://github.com/whitelok/ldbm-image-background-remover/releases/download/1.0.0/LDBMImageBackgroundRemover)

## Build

 1. Build and install [OpenCV](http://opencv.org/).
 2. Build and install [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page).
 3. Install [Cmake](https://cmake.org/).
 4. cd ${project file}
 5. mkdir build
 6. cd build
 7. cmake ..
 8. make

## References

 - Zheng Y, Kambhamettu C. Learning based digital matting[C], Computer Vision, 2009 IEEE 12th International Conference on. IEEE, 2009: 889-896.
