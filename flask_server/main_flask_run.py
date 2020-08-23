from wsgiref.simple_server import make_server

from flask import Flask, request
import cv2
import os
import numpy as np

from face_utils import face_utils_cls

app = Flask(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

face_tools = face_utils_cls()

'''
#用于图片上传
'''


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    f = request.files.get('file')
    upload_path = os.path.join("tmp/tmp." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    f.save(upload_path)
    return upload_path


'''
#用于人脸检测
'''


@app.route("/face_detect", methods=['POST', 'GET'])
def face_detect():
    '''
    #获取图片url,此处的url为request参数
    #存放的是上传好的图片的地址
    '''

    im_url = request.args.get("url")
    im_data = cv2.imread(im_url)

    print(im_url)

    x1, y1, x2, y2 = face_tools.detection_face_by_tf(im_data)

    print(x1, y1, x2, y2)
    if x1 is None:
        return "error"

    return str([x1, y1, x2, y2])


'''
#用于人脸关键点定位
'''


@app.route('/face_landmark_dlib', methods=['POST', 'GET'])
def face_landmark_dlib():
    ### 嘴巴，眼睛，casecade ###1）landmark 关键点定位，眼部关键点，粗位置，
    # 抠取，眼部关键点的回归。2）精细粒度眼部区域的回归
    # 回归模型，人脸区域， ---》 提取眼部区域，
    #
    # sudo pip3 install dlib
    '''上传图片'''
    f = request.files.get("file")
    upload_path = os.path.join("tmp/tmp_landmark." + f.filename.split(".")[-1])
    f.save(upload_path)
    ##BGR
    im_data = cv2.imread(upload_path)
    im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
    sp = im_data.shape
    rects = face_tools.face_detector_dlib(im_data, 0)
    res = []
    if len(rects) > 0:
        face = rects[0]
        shape = face_tools.face_landmark_dlib(im_data, face)
        for pt in shape.parts():
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            ptx = (pt.x - x1) * 1.0 / (x2 - x1)
            pty = (pt.y - y1) * 1.0 / (y2 - y1)
            res.append(str(ptx))
            res.append(str(pty))
            res.append(str(pt.x * 1.0 / sp[1]))
            res.append(str(pt.y * 1.0 / sp[0]))
        if res.__len__() == 136 * 2:
            res = ",".join(res)
            return res

    return "error"


@app.route('/face_landmark', methods=['POST', 'GET'])
def face_landmark():
    # 实现图片上传
    f = request.files.get('file')
    upload_path = os.path.join("tmp/tmp_landmark." + f.filename.split(".")[-1])
    f.save(upload_path)
    ##人脸检测
    im_data = cv2.imread(upload_path)
    sp = im_data.shape  ##
    x1, y1, x2, y2 = face_tools.detection_face_by_tf(im_data)
    if x1 is None:
        return "error"

    ##提取人脸区域
    y1 = int((y1 + (y2 - y1) * 0.2) * sp[0])
    x1 = int(x1 * sp[1])
    y2 = int(y2 * sp[0])
    x2 = int(x2 * sp[1])
    face_data = im_data[y1:y2, x1:x2]
    face_data = cv2.resize(face_data, (128, 128))

    landmark_val = face_tools.face_landmark_tf(face_data)

    res = []
    ##裁剪之后的人脸框中的坐标
    ##
    ## 每一个点有４个值，分别为在原图中的坐标和在人脸框中的坐标

    for i in range(0, 136, 2):
        res.append(str(landmark_val[i]))
        res.append(str(landmark_val[i + 1]))
        res.append(str((landmark_val[i] * (x2 - x1) + x1) / sp[1]))
        res.append(str((landmark_val[i + 1] * (y2 - y1) + y1) / sp[0]))

    res = ",".join(res)
    # print(res)
    return res


'''
#注册人脸
'''


@app.route('/face_register', methods=['POST', 'GET'])
def face_register():
    ##实现图片上传
    f = request.files.get('file')
    upload_path = os.path.join("tmp/tmp." + f.filename.split(".")[-1])
    f.save(upload_path)
    ##人脸检测
    im_data = cv2.imread(upload_path)
    sp = im_data.shape
    x1, y1, x2, y2 = face_tools.detection_face_by_tf(im_data)

    if x1 is None:
        print("x1 is Node!!!")
        return "fail"
    else:
        print("Registeration OK!!!")
        y1 = int(y1 * sp[0])
        x1 = int(x1 * sp[1])
        y2 = int(y2 * sp[0])
        x2 = int(x2 * sp[1])
        '''提取人脸区域'''
        face_data = im_data[y1:y2, x1:x2]
        emb1 = face_tools.face_feature(face_data)
        strr = ",".join(str(i) for i in emb1[0])
        ##写入txt
        with open("face/feature.txt", "w") as f:
            f.writelines(strr)
        f.close()
        mess = "success"
    return mess


@app.route('/face_login', methods=['POST', 'GET'])
def face_login():
    # 图片上传
    # 人脸检测
    # 人脸特征提取
    # 加载注册人脸（人脸签到，人脸数很多，加载注册人脸放在face_login,
    # 启动服务加载/采用搜索引擎/ES）
    # 同注册人脸相似性度量
    # 返回度量结果
    f = request.files.get('file')
    upload_path = os.path.join("tmp/login_tmp." + f.filename.split(".")[-1])
    f.save(upload_path)
    ##人脸检测
    im_data = cv2.imread(upload_path)
    sp = im_data.shape
    x1, y1, x2, y2 = face_tools.detection_face_by_tf(im_data)

    if x1 is None:
        print("x1 is Node!!!")
        return "fail"
    else:
        print("Login OK!!!")
        y1 = int(y1 * sp[0])
        x1 = int(x1 * sp[1])
        y2 = int(y2 * sp[0])
        x2 = int(x2 * sp[1])
        '''提取人脸区域'''
        face_data = im_data[y1:y2, x1:x2]
        emb1 = face_tools.face_feature(face_data)
        emb2 = face_tools.load_fea_from_str("face/feature.txt")
        dist = np.linalg.norm(emb1 - emb2)
        print("dist---->", dist)
        if dist < 0.3:
            return "success"
        else:
            return "fail"


@app.route('/face_attribute', methods=['POST', 'GET'])
def face_attribute():
    # 实现图片上传
    f = request.files.get('file')
    print(f)
    upload_path = os.path.join("tmp/tmp_attribute." + f.filename.split(".")[-1])
    # secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
    print(upload_path)
    f.save(upload_path)
    ##人脸检测
    im_data = cv2.imread(upload_path)
    sp = im_data.shape
    x1, y1, x2, y2 = face_tools.detection_face_by_tf(im_data)

    if x1 is None:
        return "fail"
    else:
        y1 = int(y1 * sp[0])
        x1 = int(x1 * sp[1])
        y2 = int(y2 * sp[0])
        x2 = int(x2 * sp[1])
        y1 = int(max(y1 - 0.3 * (y2 - y1), 0))
        im_data = im_data[y1:y2, x1:x2]
        im_data = cv2.resize(im_data, (128, 128))
        eye_glass, young, male, smiling = face_tools.face_attribute(im_data)
        # 训练的模型young和smiling数据打包反了
        return "{},{},{},{}".format(eye_glass[0], young[0], male[0], smiling[0])


if __name__ == '__main__':
    # app.run(host="127.0.0.1", port=90, debug=False)
    server = make_server('127.0.0.1', 90, app)
    server.serve_forever()
    app.run()
