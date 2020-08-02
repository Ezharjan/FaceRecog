import os

import cv2
import dlib
import numpy as np
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class face_utils_cls(object):
    def __init__(self):
        """
        dlib库对应的关键点模型
        """
        self.face_landmark_dlib = dlib.shape_predictor(
            "models/shape_predictor_68_face_landmarks.dat"
        )
        self.face_detector_dlib = dlib.get_frontal_face_detector()

        # set sess
        """如果使用gpu，按需分配"""
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=gpu_options,
        )
        """
        初始化人脸特征模型, 人脸检测模型，人脸关键点模型
        """

        self.face_feature_sess = tf.compat.v1.Session(
            graph=tf.Graph(), config=session_config
        )
        self.face_detection_sess = tf.compat.v1.Session(
            graph=tf.Graph(), config=session_config
        )
        self.face_landmark_sess = tf.compat.v1.Session(
            graph=tf.Graph(), config=session_config
        )
        self.face_attribute_sess = tf.compat.v1.Session(
            graph=tf.Graph(), config=session_config
        )

        self.ff_pb_path = "models/face_recognition_model.pb"
        self.init_feature_face()

        self.detect_pb_path = "models/face_detection_model.pb"
        self.init_detection_face_tf()

        self.landmark_pb_path = "models/landmark.pb"
        self.init_face_landmark_tf()

        self.attribute_pb_path = "models/face_attribute.pb"
        self.init_face_attribute()

    def init_feature_face(self):
        with self.face_feature_sess.as_default():
            with self.face_feature_sess.graph.as_default():
                with tf.compat.v1.gfile.FastGFile(self.ff_pb_path, "rb") as f:
                    graph_def = self.face_feature_sess.graph_def
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name="")
                    self.ff_images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        "input:0"
                    )
                    self.ff_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        "phase_train:0"
                    )
                    self.ff_embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        "embeddings:0"
                    )

    def init_detection_face_tf(self):
        with self.face_detection_sess.as_default():
            with self.face_detection_sess.graph.as_default():
                face_detect_od_graph_def = self.face_detection_sess.graph_def
                with tf.compat.v1.gfile.GFile(self.detect_pb_path, "rb") as fid:
                    serialized_graph = fid.read()
                    face_detect_od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(face_detect_od_graph_def, name="")
                    ops = tf.compat.v1.get_default_graph().get_operations()
                    all_tensor_names = {
                        output.name for op in ops for output in op.outputs
                    }
                    self.detection_tensor_dict = {}
                    for key in [
                        "num_detections",
                        "detection_boxes",
                        "detection_scores",
                        "detection_classes",
                    ]:
                        tensor_name = key + ":0"
                        if tensor_name in all_tensor_names:
                            self.detection_tensor_dict[
                                key
                            ] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                                tensor_name
                            )
                    self.detection_image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        "image_tensor:0"
                    )

    def init_face_landmark_tf(self):

        with self.face_landmark_sess.as_default():
            with self.face_landmark_sess.graph.as_default():
                graph_def = self.face_landmark_sess.graph_def
                with tf.compat.v1.gfile.GFile(self.landmark_pb_path, "rb") as fid:
                    serialized_graph = fid.read()
                    graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(graph_def, name="")
                    self.face_landmark_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        "fully_connected_9/Relu:0"
                    )

    def init_face_attribute(self):

        with self.face_attribute_sess.as_default():
            with self.face_attribute_sess.graph.as_default():
                graph_def = self.face_attribute_sess.graph_def
                with tf.compat.v1.gfile.GFile(self.attribute_pb_path, "rb") as fid:
                    serialized_graph = fid.read()
                    graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(graph_def, name="")
                    self.pred_eyeglasses = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        "ArgMax:0"
                    )
                    self.pred_young = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        "ArgMax_1:0"
                    )
                    self.pred_male = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        "ArgMax_2:0"
                    )
                    self.pred_smiling = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        "ArgMax_3:0"
                    )
                    self.face_attribute_image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        "Placeholder:0"
                    )

    def detection_face_by_dlib(self, im_data):
        # 调用dlib

        sp = im_data.shape
        im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2GRAY)
        rects = self.face_detector_dlib(im_data, 0)

        if len(rects) == 0:
            return None, None, None, None

        # 只取第一个人脸
        x1 = rects[0].left() * 1.0 / sp[1]
        y1 = rects[0].top() * 1.0 / sp[0]
        x2 = rects[0].right() * 1.0 / sp[1]
        y2 = rects[0].bottom() * 1.0 / sp[0]

        """
        #调整人脸区域
        """
        y1 = int(max(y1 - 0.3 * (y2 - y1), 0))

        return x1, y1, x2, y2

    def detection_face_by_tf(self, im_data):
        im_data_re = cv2.resize(im_data, (256, 256))

        print("begin ... detection")

        output_dict = self.face_detection_sess.run(
            self.detection_tensor_dict,
            feed_dict={self.detection_image_tensor: np.expand_dims(im_data_re, 0)},
        )
        print("success ... detection")

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict["num_detections"] = int(output_dict["num_detections"][0])
        output_dict["detection_classes"] = output_dict["detection_classes"][0].astype(
            np.uint8
        )
        output_dict["detection_boxes"] = output_dict["detection_boxes"][0]
        output_dict["detection_scores"] = output_dict["detection_scores"][0]

        for i in range(len(output_dict["detection_scores"])):
            if output_dict["detection_scores"][i] > 0.1:
                bbox = output_dict["detection_boxes"][i]
                y1 = bbox[0]
                x1 = bbox[1]
                y2 = bbox[2]
                x2 = bbox[3]
                return x1, y1, x2, y2

        return None, None, None, None

    # 图像数据标准化
    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def face_feature(self, face_data):
        im_data = self.prewhiten(face_data)  # 预处理
        im_data = cv2.resize(im_data, (160, 160))
        im_data1 = np.expand_dims(im_data, axis=0)
        # 人脸特征提取
        emb1 = self.face_feature_sess.run(
            self.ff_embeddings,
            feed_dict={
                self.ff_images_placeholder: im_data1,
                self.ff_train_placeholder: False,
            },
        )
        return emb1

    def face_landmark_tf(self, face_data):

        print("begin ... landmark")
        pred = self.face_landmark_sess.run(
            self.face_landmark_tensor, {"Placeholder:0": np.expand_dims(face_data, 0)}
        )
        print("success ... landmark")
        pred = pred[0]
        # cv2.imwrite("0_landmark.jpg", face_data)

        return pred

    def face_attribute(self, im_data):
        [eye_glass, young, male, smiling] = self.face_attribute_sess.run(
            [self.pred_eyeglasses, self.pred_young, self.pred_male, self.pred_smiling],
            feed_dict={self.face_attribute_image_tensor: np.expand_dims(im_data, 0)},
        )

        return eye_glass, young, male, smiling

    def load_fea_from_str(self, fea_path):
        with open(fea_path) as f:
            fea_str = f.readlines()
            f.close()
        emb2_str = fea_str[0].split(",")
        emb2 = []
        for ss in emb2_str:
            emb2.append(float(ss))
        emb2 = np.array(emb2)

        return emb2


if __name__ == "__main__":
    face_utils_cls_tool = face_utils_cls()

    im_data = cv2.imread("D:/4-th_Grade/FaceRecog/flask_server/tmp/tmp.jpg")

    x1, y1, x2, y2 = face_utils_cls_tool.detection_face_by_tf(im_data)
    sp = im_data.shape

    # 提取人脸区域
    y1 = int((y1 + (y2 - y1) * 0.2) * sp[0])
    x1 = int(x1 * sp[1])
    y2 = int(y2 * sp[0])
    x2 = int(x2 * sp[1])
    face_data = im_data[y1:y2, x1:x2]
    face_data = cv2.resize(face_data, (128, 128))

    landmark_val = face_utils_cls_tool.face_landmark_tf(face_data)
    landmark_val = ",".join(landmark_val)
    # landmark_val = ",".join(('%s' %landmark for landmark in landmark_val))
    print(landmark_val)
