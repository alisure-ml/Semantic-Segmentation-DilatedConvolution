import tensorflow as tf


class DilatedConvolution(object):

    def __init__(self, dataset, input_tensor, w_pretrained, trainable):
        self.dataset = dataset
        self.input_tensor = input_tensor
        self.w_pretrained = w_pretrained
        self.trainable = trainable

        self.softmax = self.dilated_convolution_by_pretrained()
        pass

    # 卷积
    def _conv(self, name, input, strides=list([1, 1, 1, 1]), padding="VALID", add_bias=True, apply_relu=True, atrous_rate=None):
        with tf.variable_scope(name):
            w_kernel = tf.Variable(initial_value=self.w_pretrained[name + '/kernel:0'], trainable=self.trainable)

            if atrous_rate is None:
                conv_out = tf.nn.conv2d(input, w_kernel, strides, padding)
            else:
                # 首先，对于空洞卷积，将卷积核“膨胀”：
                # 1.若进行VALID卷积，需要对输入padding atrous_rate-1个0，否则最外围的输入值就没有被卷积。
                #   所以，如果对输入不进行padding，空洞卷积的输出应该减少：
                #       reduce_size = {[k_size * atrous_rate + (atrous_rate - 1)] - 1}
                #   由于对输入padding了 2 * (atrous_rate - 1)个0，所以最终减少了 reduce_size - 2 * (atrous_rate - 1)
                # 2.若进行SAME卷积，输入和输出的形状相同。
                # 3.计算感受野：
                #   1)相对于原始输入计算：
                #       Define the receptive field of an element p in F_i+1 as the set of elements in F_0
                #       that modify the value of F_i+1(p).
                #       计算公式为：F_i+1(p) = 2 * F_i(p) - 1， F_0 = k_size
                #   2)相对与前一个输入：
                #       感受野大小为：“核的大小” - “最外围无效的区域”
                #           [k_size * atrous_rate + (atrous_rate - 1)] - 2 * (atrous_rate - 1)

                # 图像语义分割需要获取“上下文信息”，
                #   1.“不使用空洞卷积的传统卷积”通过“下采样（pooling）”来增大感受野（卷积核的大小不变，减少了数据量，
                #       有利于防止过拟合，但是损失了分辨率，丢失了一些信息），以此获取上下文信息。
                #   2.“空洞卷积”通过“膨胀卷积核”来增大感受野（参数个数没有增加，数据量没有减少，分辨率没有损失，
                #       因此，计算量增大，内存消耗增大），以此获取上下文信息。
                conv_out = tf.nn.atrous_conv2d(input, w_kernel, atrous_rate, padding)

            if add_bias:
                w_bias = tf.Variable(initial_value=self.w_pretrained[name + '/bias:0'], trainable=self.trainable)
                conv_out = tf.nn.bias_add(conv_out, w_bias)

            if apply_relu:
                conv_out = tf.nn.relu(conv_out)

        return conv_out

    def dilated_convolution_by_pretrained(self):
        # Check on dataset name
        if self.dataset not in ['cityscapes', 'camvid']:
            raise ValueError('Dataset "{}" not supported.'.format(self.dataset))
        else:
            h = self.input_tensor  # [1396, 1396, 3]
            h = self._conv('conv1_1', h)  # [1394, 1394, 64]
            h = self._conv('conv1_2', h)  # [1392, 1392, 64]
            h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool1')  # [696, 696, 64]

            h = self._conv('conv2_1', h)  # [694, 694, 128]
            h = self._conv('conv2_2', h)  # [692, 692, 128]
            h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2')  # [346, 346, 128]

            h = self._conv('conv3_1', h)  # [344, 344, 256]
            h = self._conv('conv3_2', h)  # [342, 342, 256]
            h = self._conv('conv3_3', h)  # [340, 340, 256]
            h = tf.layers.max_pooling2d(h, pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool3')  # [170, 170, 256]

            h = self._conv('conv4_1', h)  # [168, 168, 512]
            h = self._conv('conv4_2', h)  # [166, 166, 512]
            h = self._conv('conv4_3', h)  # [164, 164, 512]

            h = self._conv('conv5_1', h, atrous_rate=2)  # [160, 160, 512]
            h = self._conv('conv5_2', h, atrous_rate=2)  # [156, 156, 512]
            h = self._conv('conv5_3', h, atrous_rate=2)  # [152, 152, 512]
            h = self._conv('fc6', h, atrous_rate=4)  # [128, 128, 4096] (k=7)

            h = tf.layers.dropout(h, rate=0.5, name='drop6')
            h = self._conv('fc7', h)  # [128, 128, 4096]
            h = tf.layers.dropout(h, rate=0.5, name='drop7')
            h = self._conv('final', h)  # [128, 128, 19]
            # 上面属于Front-End

            # 下面开始Context network architecture
            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_1')  # [130, 130, 19]
            h = self._conv('ctx_conv1_1', h)  # [128, 128, 19] receptive field=[3, 3]
            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad1_2')  # [130, 130, 19]
            h = self._conv('ctx_conv1_2', h)  # [128, 128, 19] rf=[5, 5]

            h = tf.pad(h, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT', name='ctx_pad2_1')  # [132, 132, 19]
            h = self._conv('ctx_conv2_1', h, atrous_rate=2)  # [128, 128, 19] rf=[9, 9]

            h = tf.pad(h, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT', name='ctx_pad3_1')  # [136, 136, 19]
            h = self._conv('ctx_conv3_1', h, atrous_rate=4)  # [128, 128, 19] rf=[17, 17]

            h = tf.pad(h, [[0, 0], [8, 8], [8, 8], [0, 0]], mode='CONSTANT', name='ctx_pad4_1')  # [144, 144, 19]
            h = self._conv('ctx_conv4_1', h, atrous_rate=8)  # [128, 128, 19] rf=[33, 33]

            h = tf.pad(h, [[0, 0], [16, 16], [16, 16], [0, 0]], mode='CONSTANT', name='ctx_pad5_1')  # [160, 160, 19]
            h = self._conv('ctx_conv5_1', h, atrous_rate=16)  # [128, 128, 19] rf=[65, 65]

            if self.dataset == 'cityscapes':
                h = tf.pad(h, [[0, 0], [32, 32], [32, 32], [0, 0]], mode='CONSTANT', name='ctx_pad6_1')  # [192, 192, 19]
                h = self._conv('ctx_conv6_1', h, atrous_rate=32)  # [128, 128, 19]  rf=[129, 129]

                h = tf.pad(h, [[0, 0], [64, 64], [64, 64], [0, 0]], mode='CONSTANT', name='ctx_pad7_1')  # [256, 256, 19]
                h = self._conv('ctx_conv7_1', h, atrous_rate=64)  # [128, 128, 19]  rf=[257, 257]
                pass

            h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT', name='ctx_pad_fc1')  # [130, 130, 19]
            h = self._conv('ctx_fc1', h)  # [128, 128, 19]
            h = self._conv('ctx_final', h, padding='VALID', add_bias=True, apply_relu=False)  # [128, 128, 19]

            if self.dataset == 'cityscapes':
                h = tf.image.resize_bilinear(h, size=(1024, 1024))  # [1024, 1024, 19]
                logits = self._conv('ctx_upsample', h, padding='SAME', add_bias=False, apply_relu=True)  # [1024, 1024, 19]
            else:
                logits = h  # [128, 128, 19]

            softmax = tf.nn.softmax(logits, dim=3, name='softmax')

        return softmax

    pass
