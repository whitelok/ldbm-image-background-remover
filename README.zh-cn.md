# Learning Based Digital Matting 前景图像提取算法C/C++版

*其他语言版本: [English](README.md), [简体中文](README.zh-cn.md).*

## 论文

```latex
@InProceedings{ZhengICCV09,
  author = {Yuanjie Zheng and Chandra Kambhamettu},
  title = {Learning Based Digital Matting},
  booktitle = {The 20th IEEE International Conference on Computer Vision},
  year = {2009},
  month = {September--October}
}
```

## 演示

如图所示，左边第一张为需要进行前景图像分割的源图片，左边第二张图为标记图片。其中标记图片中的白色区域为确定的前景图像区域，灰色区域为前景图像边缘区域，黑色区域为背景区域。当输入源图片和标记图片之后，根据算法可得出右边前景图像提取后的图片，其中白色像素点为前景图像的像素点，黑色像素点为背景图像的像素点。

![图1](res/img/demo_1.png)

![图2](res/img/demo_2.png)

## 项目结构

-bin // 二进制可执行文件位置

-data // 用于测试图片资源

-res // README文件演示用图片

-src // 源代码
