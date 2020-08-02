# Face Recognition --- Flask Server

重新整理了基于Python - Flask的服务端代码，主要函数封装在了face_utils.py文件中
- 模型路径：
- 链接: https://pan.baidu.com/s/1jSs-pEF-UbDFOFDZLJsCbw 提取码: pca3 。
- 包含模型：facedetection，facelandmark，facerecognition，faceattribute。
- 客户端基于微信小程序开发，在本总项目文件夹的wx_client文件夹中，使用微信开发者工具即可打开运行。



备注:原代码在版本上具有很多不兼容性问题，其中一处涉及 --- TypeError: required field "type_ignores" missing from Module --- 的问题， 鉴于此，本人参考的具体解决方式来源于https://www.jianshu.com/p/95588bf4e63d 。 除此之外仍有许多地方会弹出兼容性问题，如果是具体的某个函数，则可以搜索这个函数的在新版本上的用法，或者直接在原来tf的调用方式后面加上compat.v1.即可解决，本项目中使用的方案也是这种解决方式。