import tensorflow as tf
import pickle
import cv2
import os
import numpy as np
from DilatedConvolution import DilatedConvolution
from DataConfig import Config


class InferenceRunner(object):

    def __init__(self, dataset, input_file, output_path):
        self.dataset = dataset
        self.input_file = input_file
        self.output_file = os.path.join(self.new_dir(output_path), os.path.split(input_file)[1])

        # 创建checkpoint目录
        self.checkpoint = self.new_dir(os.path.join("checkpoint", 'dilated_' + self.dataset))
        self.checkpoint_file = os.path.join(self.checkpoint, "dilated")
        self.checkpoint_file_meta = self.checkpoint_file + ".meta"

        # 转换模型
        if not os.path.exists(self.checkpoint_file_meta):
            self.pre_trained_pickle_to_checkpoint()
        print("Model has existed ...")

        # 测试
        self.inference()
        pass

    def inference(self):
        print('Begin to inference ...')
        # Restore both graph and weights from TF checkpoint
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            # 从checkpoint中导入模型
            saver = tf.train.import_meta_graph(self.checkpoint_file_meta)
            saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint))

            graph = tf.get_default_graph()

            # 输入
            input_image = cv2.imread(self.input_file)
            input_tensor = graph.get_tensor_by_name('input_placeholder:0')

            # 结果
            softmax = graph.get_tensor_by_name('softmax:0')
            softmax = tf.reshape(softmax, shape=(1,) + Config[self.dataset]['output_shape'])

            # 预测
            predicted_image = self.predict(input_image, input_tensor, softmax, self.dataset, sess)

            # 保存
            print('Save result in {}'.format(self.output_file))
            cv2.imwrite(self.output_file, cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB))
        pass

    def pre_trained_pickle_to_checkpoint(self):
        print('Loading pre-trained weights...')
        with open(Config[self.dataset]['weights_file'], 'rb') as f:
            w_pretrained = pickle.load(f)
        print('Loading pre-trained weights Done.')

        input_h, input_w, input_c = Config[self.dataset]['input_shape']
        input_tensor = tf.placeholder(tf.float32, shape=(None, input_h, input_w, input_c), name='input_placeholder')
        DilatedConvolution(self.dataset, input_tensor, w_pretrained, trainable=False)

        # 保存模型
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)).save(sess, self.checkpoint_file)
        pass

    # this function is the same as the one in the original repository
    # basically it performs upsampling for datasets having zoom > 1
    @staticmethod
    def interp_map(prob, zoom, width, height):
        channels = prob.shape[2]
        zoom_prob = np.zeros((height, width, channels), dtype=np.float32)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    r0 = h // zoom
                    r1 = r0 + 1
                    c0 = w // zoom
                    c1 = c0 + 1
                    rt = float(h) / zoom - r0
                    ct = float(w) / zoom - c0
                    v0 = rt * prob[r1, c0, c] + (1 - rt) * prob[r0, c0, c]
                    v1 = rt * prob[r1, c1, c] + (1 - rt) * prob[r0, c1, c]
                    zoom_prob[h, w, c] = (1 - ct) * v0 + ct * v1
        return zoom_prob

    # predict function, mostly reported as it was in the original repo
    def predict(self, image, input_tensor, model, ds, sess):

        image = image.astype(np.float32) - Config[ds]['mean_pixel']

        input_height, input_width, num_channels = Config[ds]['input_shape']
        model_in = np.zeros((1, input_height, input_width, num_channels), dtype=np.float32)

        conv_margin = Config[ds]['conv_margin']
        image_size = image.shape
        output_height = input_height - 2 * conv_margin
        output_width = input_width - 2 * conv_margin
        image = cv2.copyMakeBorder(image, conv_margin, conv_margin, conv_margin, conv_margin, cv2.BORDER_REFLECT_101)

        num_tiles_h = image_size[0] // output_height + (1 if image_size[0] % output_height else 0)
        num_tiles_w = image_size[1] // output_width + (1 if image_size[1] % output_width else 0)

        row_prediction = []
        for h in range(num_tiles_h):
            col_prediction = []
            for w in range(num_tiles_w):
                offset = [output_height * h, output_width * w]
                tile = image[offset[0]:offset[0] + input_height, offset[1]:offset[1] + input_width, :]
                margin = [0, input_height - tile.shape[0], 0, input_width - tile.shape[1]]
                tile = cv2.copyMakeBorder(tile, margin[0], margin[1], margin[2], margin[3], cv2.BORDER_REFLECT_101)

                model_in[0] = tile
                prob = sess.run(model, feed_dict={input_tensor: tile[None, ...]})[0]
                col_prediction.append(prob)

            col_prediction = np.concatenate(col_prediction, axis=1)  # previously axis=2
            row_prediction.append(col_prediction)
        prob = np.concatenate(row_prediction, axis=0)
        if Config[ds]['zoom'] > 1:
            prob = self.interp_map(prob, Config[ds]['zoom'], image_size[1], image_size[0])

        prediction = np.argmax(prob, axis=2)
        color_image = Config[ds]['palette'][prediction.ravel()].reshape(image_size)

        return color_image

    @staticmethod
    def new_dir(path_name):
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        return path_name

    pass


if __name__ == '__main__':

    # Choose between 'cityscapes' and 'camvid'
    InferenceRunner(dataset="camvid", input_file="input/test_my.png", output_path="result")

    pass
