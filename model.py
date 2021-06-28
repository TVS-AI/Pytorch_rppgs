import torch


class appearance_model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        # Appearance model
        super().__init__()
        self.a_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        self.a_batch_Normalization1 = torch.nn.BatchNorm2d(32)
        self.a_conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_batch_Normalization2 = torch.nn.BatchNorm2d(32)
        self.a_dropout1 = torch.nn.Dropout2d(p=0.50)
        # Attention mask1
        self.attention_mask1 = attention_block(32)
        self.a_avg1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.a_conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.a_Batch_Normalization3 = torch.nn.BatchNorm2d(64)
        self.a_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.a_Batch_Normalization4 = torch.nn.BatchNorm2d(64)
        self.a_dropout2 = torch.nn.Dropout2d(p=0.50)
        # Attention mask2
        self.attention_mask2 = attention_block(64)

    def forward(self, inputs):
        # Convlution layer
        A1 = torch.tanh(self.a_batch_Normalization1(self.a_conv1(inputs)))
        A2 = torch.tanh(self.a_batch_Normalization2(self.a_conv2(A1)))
        A3 = self.a_dropout1(A2)
        # Calculate Mask1
        M1 = self.attention_mask1(A3)
        # Pooling
        A4 = self.a_avg1(A3)
        # Convolution layer
        A5 = torch.tanh(self.a_Batch_Normalization3(self.a_conv3(A4)))
        A6 = torch.tanh(self.a_Batch_Normalization4(self.a_conv4(A5)))
        A7 = self.a_dropout2(A6)
        # Calculate Mask2
        M2 = self.attention_mask2(A7)

        return M1, M2


class motion_model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, model):
        super().__init__()
        # Motion model
        self.m_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        self.m_batch_Normalization1 = torch.nn.BatchNorm2d(32)
        self.m_conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=1, padding=1)
        self.m_batch_Normalization2 = torch.nn.BatchNorm2d(32)
        self.m_dropout1 = torch.nn.Dropout2d(p=0.50)

        self.m_avg1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_conv3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=64, kernel_size=kernel_size, stride=1,
                                       padding=1)
        self.m_batch_Normalization3 = torch.nn.BatchNorm2d(64)
        self.m_conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1, padding=1)
        self.m_batch_Normalization4 = torch.nn.BatchNorm2d(64)
        self.m_dropout2 = torch.nn.Dropout2d(p=0.50)
        self.m_avg2 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs, mask1, mask2):
        M1 = torch.tanh(self.m_batch_Normalization1(self.m_conv1(inputs)))
        M2 = self.m_batch_Normalization2(self.m_conv2(M1))
        # element wise multiplication Mask1
        g1 = torch.tanh(torch.mul(1 * mask1, M2))
        M3 = self.m_dropout1(g1)
        # pooling
        M4 = self.m_avg1(M3)
        M5 = torch.tanh(self.m_batch_Normalization3(self.m_conv3(M4)))
        M6 = self.m_batch_Normalization4(self.m_conv4(M5))
        # element wise multiplication Mask2
        g2 = torch.tanh(torch.mul(1 * mask2, M6))
        M7 = self.m_dropout2(g2)
        out = self.m_avg2(M7)

        return out


class attention_block(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = torch.nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        mask = self.attention(input)
        mask = torch.sigmoid(mask)
        B, _, H, W = input.shape
        norm = 2 * torch.norm(mask, p=1, dim=(1, 2, 3))
        norm = norm.reshape(B, 1, 1, 1)
        mask = torch.div(mask * H * W, norm)
        return mask


class fc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f_drop1 = torch.nn.Dropout(0.25)
        self.f_linear1 = torch.nn.Linear(64 * 9 * 9, 128, bias=True)
        self.f_linear2 = torch.nn.Linear(128, 1, bias=True)

    def forward(self, input):
        f1 = torch.flatten(input, start_dim=1)
        f2 = self.f_drop1(f1)
        f3 = torch.tanh(self.f_linear1(f2))
        f4 = self.f_linear2(f3)
        return f4


class TSM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input, n_frame=4, fold_div=3):
        n_frame = 4
        B, C, H, W = input.shape
        input = input.view(-1, n_frame, H, W, C)
        fold = C // fold_div
        last_fold = C - (fold_div - 1) * fold
        out1, out2, out3 = torch.split(input, [fold, fold, last_fold], -1)

        padding1 = torch.zeros_like(out1)
        padding1 = padding1[:, -1, :, :, :]
        padding1 = torch.unsqueeze(padding1, 1)
        _, out1 = torch.split(out1, [1, n_frame - 1], 1)
        out1 = torch.cat((out1, padding1), 1)

        padding2 = torch.zeros_like(out2)
        padding2 = padding2[:, 0, :, :, :]
        padding2 = torch.unsqueeze(padding2, 1)
        out2, _ = torch.split(out2, [n_frame - 1, 1], 1)
        out2 = torch.cat((padding2, out2), 1)

        out = torch.cat((out1, out2, out3), -1)
        out = out.view([-1, C, H, W])

        return out


class TSM_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.tsm1 = TSM()
        self.t_conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       padding=1)

    def forward(self, input, n_frame=2, fold_div=3):
        t = self.tsm1(input, n_frame, fold_div)
        t = self.t_conv1(t)
        return t


class motion_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, model):
        super().__init__()
        self.model = model

        if self.model.find("TS") is not -1:
            self.m_tsm1 = TSM_block(in_channels, out_channels, kernel_size)
            self.m_tsm2 = TSM_block(out_channels, out_channels, kernel_size)
        else:
            self.m_conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
            self.m_batch1 = torch.nn.BatchNorm2d(out_channels)
            self.m_conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
            self.m_batch2 = torch.nn.BatchNorm2d(out_channels)
        self.m_drop3 = torch.nn.Dropout2d(0.5)
        self.m_avg3 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs, mask):
        if self.model.find("TS") is not -1:
            m = self.m_tsm1(inputs)
            m = torch.tanh(m)
            m = self.m_tsm2(m)
        else:
            m = self.m_conv1(inputs)
            m = self.m_batch1(m)
            m = torch.tanh(m)

            m = self.m_conv2(m)
            m = self.m_batch2(m)

        m = torch.mul(m, mask)
        m = torch.tanh(m)

        m = self.m_drop3(m)
        m = self.m_avg3(m)
        return m


class model(torch.nn.Module):
    def __init__(self, model="CAN"):
        super().__init__()
        self.model = model
        self.in_channels = 3
        self.out_channels = 32
        self.kernel_size = 3

        self.appearance_model = appearance_model(in_channels=self.in_channels, out_channels=self.out_channels,
                                                 kernel_size=self.kernel_size)
        self.motion_model = motion_model(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=self.kernel_size, model=model)
        self.fully = fc()

    def forward(self, appearance_input, motion_input):
        attention_mask1, attention_mask2 = self.appearance_model(appearance_input)
        motion_output = self.motion_model(motion_input, attention_mask1, attention_mask2)
        out = self.fully(motion_output)

        return out, attention_mask1, attention_mask2


class Denoising_LSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(input_size=4, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(128, 1)
        self.timedist = TimeDistributed(torch.nn.Linear(1, 1))

    def forward(self, noise):
        # layer1 = self.lstm1(noise)
        (LSTM_layer1, _) = self.lstm1(noise)
        # RepeatVector - pytorch
        LSTM_layer1 = LSTM_layer1[:, -1, :].unsqueeze(1).repeat(1, 60, 1)
        # layer2 = self.lstm2(layer1)
        (LSTM_layer2, _) = self.lstm2(LSTM_layer1)
        # fc_layer = torch.tanh(self.fc(LSTM_layer2))
        fully = torch.tanh(self.fc(LSTM_layer2))
        # output = TimeDistributed(torch.tanh(self.fc(LSTM_layer2)))
        output = self.timedist(fully)
        return output


class TimeDistributed(torch.nn.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y
