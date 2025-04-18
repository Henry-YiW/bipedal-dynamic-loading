import torch
import torch.nn as nn


class Improved1DConvEncoder(nn.Module):
    def __init__(self, input_dim=48, embedding_dim=64, z_dim=10, input_seq_length=20):
        super(Improved1DConvEncoder, self).__init__()

        # 输入嵌入MLP
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim), nn.ReLU()
        )

        # 一维卷积层
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(256)

        # 计算卷积后的长度
        conv_output_length = self._get_conv_output_length(input_seq_length, [4, 3], [2, 1])

        # 线性投影层
        self.fc1 = nn.Linear(256 * conv_output_length, 128)
        self.fc2 = nn.Linear(128, z_dim)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_conv_output_length(self, length, kernel_sizes, strides):
        for k, s in zip(kernel_sizes, strides):
            length = (length - k) // s + 1
        return length

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape

        # 将输入嵌入
        x = x.view(-1, input_dim)
        embeddings = self.input_embedding(x)

        # 恢复到(batch_size, seq_length, embedding_dim)
        embeddings = embeddings.view(batch_size, seq_length, -1)

        # 转置为(batch_size, embedding_dim, seq_length)
        embeddings = embeddings.transpose(1, 2)

        # 通过卷积层
        x = self.conv1(embeddings)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # 添加 Dropout 层
        x = self.dropout(x)

        # 展平并通过线性投影层
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        z = self.fc2(x)

        return z


class SimplifiedAdaptationModule1DConvEncoderB(nn.Module):
    def __init__(self, input_dim=48, embedding_dim=32, z_dim=8, input_seq_length=20):
        super(SimplifiedAdaptationModule1DConvEncoderB, self).__init__()

        # 输入嵌入MLP
        self.input_embedding = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.ReLU())

        # 一维卷积层
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2)

        # 计算卷积后的长度
        conv_output_length = self._get_conv_output_length(input_seq_length, [3, 3], [2, 2])

        # 线性投影层
        self.fc1 = nn.Linear(128 * conv_output_length, z_dim)

    def _get_conv_output_length(self, input_length, kernel_sizes, strides):
        length = input_length
        for k, s in zip(kernel_sizes, strides):
            length = (length - (k - 1) - 1) // s + 1
        return length

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape

        # 将输入嵌入
        x = x.view(-1, input_dim)
        embeddings = self.input_embedding(x)

        # 恢复到(batch_size, seq_length, embedding_dim)
        embeddings = embeddings.view(batch_size, seq_length, -1)

        # 转置为(batch_size, embedding_dim, seq_length)
        embeddings = embeddings.transpose(1, 2)

        # 通过卷积层
        x = self.conv1(embeddings)
        x = self.conv2(x)

        # 展平并通过线性投影层
        x = x.view(batch_size, -1)
        z = self.fc1(x)

        return z


class AdaptationModule1DConvEncoder(nn.Module):
    def __init__(self, input_dim=48, embedding_dim=64, z_dim=10):
        super(AdaptationModule1DConvEncoder, self).__init__()

        # 输入嵌入MLP
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim), nn.ReLU()
        )

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 一维卷积层
        self.conv1 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=4, stride=2)
        self.conv2 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=2, stride=1)

        # 计算卷积后的长度
        conv_output_length = self._get_conv_output_length(20, [4, 3, 2], [2, 1, 1])

        # 线性投影层
        self.fc = nn.Linear(embedding_dim * conv_output_length, z_dim)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape

        # 将输入嵌入
        x = x.view(-1, input_dim)
        embeddings = self.input_embedding(x)

        # 恢复到(batch_size, seq_length, embedding_dim)
        embeddings = embeddings.view(batch_size, seq_length, -1)

        # 转置为(batch_size, embedding_dim, seq_length)
        embeddings = embeddings.transpose(1, 2)

        # 通过卷积层
        x = self.conv1(embeddings)
        x = self.conv2(x)
        x = self.conv3(x)

        # 添加 Dropout 层
        x = self.dropout(x)

        # 展平并通过线性投影层
        x = x.view(batch_size, -1)
        z = self.fc(x)

        return z

    def _get_conv_output_length(self, input_length, kernel_sizes, strides):
        length = input_length
        for k, s in zip(kernel_sizes, strides):
            length = (length - k) // s + 1
        return length

    def _get_conv_output_length(self, input_length, kernel_sizes, strides):
        length = input_length
        for k, s in zip(kernel_sizes, strides):
            length = (length - k) // s + 1
        return length


class LargeAdaptationModule1DConvEncoder(nn.Module):
    def __init__(self, input_dim=48, embedding_dim=64, z_dim=10, input_seq_length=40):
        super(LargeAdaptationModule1DConvEncoder, self).__init__()

        # 输入嵌入MLP
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim), nn.ReLU()
        )

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 增加一维卷积层的数量和神经元数量
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(512, 512, kernel_size=2, stride=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=2, stride=1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.conv6 = nn.Conv1d(1024, 1024, kernel_size=2, stride=1)
        self.bn6 = nn.BatchNorm1d(1024)

        # 计算卷积后的长度
        conv_output_length = self._get_conv_output_length(
            input_seq_length, [4, 3, 2, 2, 2, 2], [2, 1, 1, 1, 1, 1]
        )

        # 增加全连接层的数量和神经元数量
        self.fc1 = nn.Linear(1024 * conv_output_length, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, z_dim)

    def _get_conv_output_length(self, input_length, kernel_sizes, strides):
        length = input_length
        for k, s in zip(kernel_sizes, strides):
            length = (length - k) // s + 1
        return length

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape

        # 将输入嵌入
        x = x.view(-1, input_dim)
        embeddings = self.input_embedding(x)

        # 恢复到(batch_size, seq_length, embedding_dim)
        embeddings = embeddings.view(batch_size, seq_length, -1)

        # 转置为(batch_size, embedding_dim, seq_length)
        embeddings = embeddings.transpose(1, 2)

        # 通过卷积层
        x = self.conv1(embeddings)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = torch.relu(x)

        # 添加 Dropout 层
        x = self.dropout(x)

        # 展平并通过线性投影层
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        z = self.fc4(x)

        return z


class AdaptationModule1DConvEncoderBB(nn.Module):
    def __init__(
        self, input_dim=48, embedding_dim=128, z_dim=10, input_seq_length=20  # 增加embedding维度  # 增加z维度
    ):  # 输入序列长度改为40
        super(AdaptationModule1DConvEncoderB, self).__init__()

        # 输入嵌入MLP
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim), nn.ReLU()
        )

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 一维卷积层
        self.conv1 = nn.Conv1d(embedding_dim, 256, kernel_size=4, stride=2)  # 增加卷积核数量
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(512, 1024, kernel_size=2, stride=1)
        self.conv4 = nn.Conv1d(1024, 1024, kernel_size=2, stride=1)

        # 计算卷积后的长度
        conv_output_length = self._get_conv_output_length(input_seq_length, [4, 3, 2, 2], [2, 1, 1, 1])

        # 线性投影层
        self.fc1 = nn.Linear(1024 * conv_output_length, 512)  # 增加线性层的维度
        self.fc2 = nn.Linear(512, z_dim)

    def _get_conv_output_length(self, input_length, kernel_sizes, strides):
        length = input_length
        for kernel_size, stride in zip(kernel_sizes, strides):
            length = (length - (kernel_size - 1) - 1) // stride + 1
        return length

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape

        # 将输入嵌入
        x = x.view(-1, input_dim)
        embeddings = self.input_embedding(x)

        # 恢复到(batch_size, seq_length, embedding_dim)
        embeddings = embeddings.view(batch_size, seq_length, -1)

        # 转置为(batch_size, embedding_dim, seq_length)
        embeddings = embeddings.transpose(1, 2)

        # 通过卷积层
        x = self.conv1(embeddings)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 添加 Dropout 层
        x = self.dropout(x)

        # 展平并通过线性投影层
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        z = self.fc2(x)

        return z


class AdaptationModule1DConvEncoderB(nn.Module):
    def __init__(
        self, input_dim=48, embedding_dim=128, z_dim=32, input_seq_length=20  # 增加embedding维度  # 增加z维度
    ):  # 输入序列长度改为40
        super(AdaptationModule1DConvEncoderB, self).__init__()

        # 输入嵌入MLP
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim), nn.ReLU()
        )

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 一维卷积层
        self.conv1 = nn.Conv1d(embedding_dim, 256, kernel_size=4, stride=2)  # 增加卷积核数量
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(512, 1024, kernel_size=2, stride=1)
        self.conv4 = nn.Conv1d(1024, 1024, kernel_size=2, stride=1)

        # 计算卷积后的长度
        conv_output_length = self._get_conv_output_length(input_seq_length, [4, 3, 2, 2], [2, 1, 1, 1])

        # 线性投影层
        self.fc1 = nn.Linear(1024 * conv_output_length, 512)  # 增加线性层的维度
        self.fc2 = nn.Linear(512, z_dim)

    def _get_conv_output_length(self, input_length, kernel_sizes, strides):
        length = input_length
        for kernel_size, stride in zip(kernel_sizes, strides):
            length = (length - (kernel_size - 1) - 1) // stride + 1
        return length

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape

        # 将输入嵌入
        x = x.view(-1, input_dim)
        embeddings = self.input_embedding(x)

        # 恢复到(batch_size, seq_length, embedding_dim)
        embeddings = embeddings.view(batch_size, seq_length, -1)

        # 转置为(batch_size, embedding_dim, seq_length)
        embeddings = embeddings.transpose(1, 2)

        # 通过卷积层
        x = self.conv1(embeddings)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 添加 Dropout 层
        x = self.dropout(x)

        # 展平并通过线性投影层
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        z = self.fc2(x)

        return z


class EstModule1DConvEncoderB(nn.Module):
    def __init__(
        self, input_dim=48, embedding_dim=128, z_dim=32, input_seq_length=20  # 增加embedding维度  # 增加z维度
    ):  # 输入序列长度改为40
        super(EstModule1DConvEncoderB, self).__init__()

        # 输入嵌入MLP
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim), nn.ReLU()
        )

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 一维卷积层
        self.conv1 = nn.Conv1d(embedding_dim, 256, kernel_size=4, stride=2)  # 增加卷积核数量
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(512, 1024, kernel_size=2, stride=1)
        self.conv4 = nn.Conv1d(1024, 1024, kernel_size=2, stride=1)

        # 计算卷积后的长度
        conv_output_length = self._get_conv_output_length(input_seq_length, [4, 3, 2, 2], [2, 1, 1, 1])

        # 线性投影层及批归一化
        self.fc1 = nn.Linear(1024 * conv_output_length, 512)  # 增加线性层的维度
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, z_dim)

        # 激活函数
        self.relu = nn.ReLU()

    def _get_conv_output_length(self, input_length, kernel_sizes, strides):
        length = input_length
        for kernel_size, stride in zip(kernel_sizes, strides):
            length = (length - (kernel_size - 1) - 1) // stride + 1
        return length

    def forward(self, x):
        batch_size, seq_length, input_dim = x.shape

        # 将输入嵌入
        x = x.view(-1, input_dim)
        embeddings = self.input_embedding(x)

        # 恢复到(batch_size, seq_length, embedding_dim)
        embeddings = embeddings.view(batch_size, seq_length, -1)

        # 转置为(batch_size, embedding_dim, seq_length)
        embeddings = embeddings.transpose(1, 2)

        # 通过卷积层
        x = self.conv1(embeddings)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # 添加 Dropout 层
        x = self.dropout(x)

        # 展平并通过线性投影层
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = self.relu(x)

        z = self.fc4(x)

        return z
