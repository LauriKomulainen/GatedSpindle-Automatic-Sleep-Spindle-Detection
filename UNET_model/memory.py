# UNET_model/memory.py

import torch
import torch.nn as nn
import logging
from utils.logger import setup_logging
setup_logging("training.log")
log = logging.getLogger(__name__)

class ConvLSTMCell(nn.Module):
    """
    Yksinkertainen ConvLSTM-solu, joka on LSTM:n ydinosa,
    mutta käyttää konvoluutioita lineaaristen kerrosten sijaan.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Tarvitsemme 4 porttia (input, forget, output, cell)
        # joten teemme yhden ison konvoluutiokerroksen tehokkuuden vuoksi
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Yhdistä syöte ja edellinen piilotila
        combined = torch.cat([input_tensor, h_cur], dim=1)  # (B, C_in + C_hidden, H, W)

        combined_conv = self.conv(combined)

        # Jaa 4 porttiin
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class BiConvLSTM(nn.Module):
    """
    Kaksisuuntainen (Bidirectional) ConvLSTM-kerros.
    Tämä ottaa kuvasekvenssin (B, S, C, H, W) ja antaa ulos
    yhdistetyn piirrekartan (B, 2*C_hidden, H, W).
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1, bias=True):
        super(BiConvLSTM, self).__init__()

        # Tässä yksinkertaistuksessa käytämme vain yhtä kerrosta (num_layers=1)
        # 2-kerroksinen BiLSTM vaatisi huomattavasti monimutkaisemman koodin
        if num_layers > 1:
            log.warning("This BiConvLSTM implementation only supports num_layers=1.")

        self.hidden_dim = hidden_dim
        self.forward_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)
        self.backward_cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)

    def forward(self, x):
        """
        Syöte x: (B, S, C, H, W)
        B = Batch size
        S = Sequence length (esim. 3)
        C = Channels (esim. 1024)
        H, W = Piirrekartan koko
        """
        b, s, _, h, w = x.size()

        # Alusta piilotilat
        h_f, c_f = self.forward_cell.init_hidden(b, (h, w))
        h_b, c_b = self.backward_cell.init_hidden(b, (h, w))

        # Eteenpäin-ajo
        for t in range(s):
            h_f, c_f = self.forward_cell(input_tensor=x[:, t, :, :, :],
                                         cur_state=[h_f, c_f])

        # Taaksepäin-ajo
        for t in reversed(range(s)):
            h_b, c_b = self.backward_cell(input_tensor=x[:, t, :, :, :],
                                          cur_state=[h_b, c_b])

        # Yhdistä viimeiset piilotilat (eteenpäin ja taaksepäin)
        output = torch.cat((h_f, h_b), dim=1)  # (B, 2 * C_hidden, H, W)
        return output