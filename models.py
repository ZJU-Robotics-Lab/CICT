import torch.nn as nn
import torch
from torch.autograd import Variable


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           C-LSTM
##############################



class CLSTMCell(nn.Module):

    # Constructor
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias=True):
        super(CLSTMCell, self).__init__()

        #assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(self.input_channels + self.hidden_channels,
                            self.num_features * self.hidden_channels,
                            self.kernel_size,
                            1,
                            self.padding)]
        self.model = nn.Sequential(*layers)

    # Forward propogation formulation
    def forward(self, x, h, c):
        combined = torch.cat((x, h), dim=1)
        A = self.model(combined)

        # NOTE: A? = xz * Wx? + hz-1 * Wh? + b? where * is convolution
        (Ai, Af, Ao, Ag) = torch.split(A,
                                       A.size()[1] // self.num_features,
                                       dim=1)

        i = torch.sigmoid(Ai)     # input gate
        f = torch.sigmoid(Af)     # forget gate
        o = torch.sigmoid(Ao)     # output gate
        g = torch.tanh(Ag)
        c = c * f + i * g           # cell activation state
        h = o * torch.tanh(c)     # cell hidden state

        return h, c

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        try:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).cuda(),
               Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).cuda())
        except:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])),
                    Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])))


''' Class CLSTM.
    This represents a series of CLSTM nodes (one direction)
'''


class CLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True):
        super(CLSTM, self).__init__()

        # store stuff
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)

        self.bias = bias
        self.all_layers = []

        # create a node for each layer in the CLSTM
        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = CLSTMCell(self.input_channels[layer],
                             self.hidden_channels[layer],
                             self.kernel_size,
                             self.bias)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    # Forward propogation
    # x --> BatchSize x NumSteps x NumChannels x Height x Width
    #       BatchSize x 2 x 64 x 240 x 240
    def forward(self, x):
        bsize, steps, _, height, width = x.size()
        internal_state = []
        outputs = []
        for step in range(steps):
            input =x[:, step, :, :, :]
            #input = torch.squeeze(x[:, step, :, :, :], dim=1)
            for layer in range(self.num_layers):
                # populate hidden states for all layers
                if step == 0:
                    (h, c) = CLSTMCell.init_hidden(bsize,
                                                   self.hidden_channels[layer],
                                                   (height, width))
                    internal_state.append((h, c))

                # do forward
                name = 'cell{}'.format(layer)
                (h, c) = internal_state[layer]

                input, c = getattr(self, name)(
                    input, h, c)  # forward propogation call
                internal_state[layer] = (input, c)

            outputs.append(input)

        #for i in range(len(outputs)):
        #    print(outputs[i].size())
        return outputs


class CLSTM_online(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True):
        super(CLSTM_online, self).__init__()

        # store stuff
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)

        self.bias = bias
        self.all_layers = []

        # create a node for each layer in the CLSTM
        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = CLSTMCell(self.input_channels[layer],
                             self.hidden_channels[layer],
                             self.kernel_size,
                             self.bias)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    # Forward propogation
    # x --> BatchSize x NumSteps x NumChannels x Height x Width
    #       BatchSize x 2 x 64 x 240 x 240
    def forward(self, x, h0, c0):
        bsize, steps, _, height, width = x.size()
        internal_state = []
        outputs = []
        for step in range(steps):
            input =x[:, step, :, :, :]
            for layer in range(self.num_layers):
                # populate hidden states for all layers
                if step == 0:
                    h = h0
                    c = c0
                    internal_state.append((h, c))

                # do forward
                name = 'cell{}'.format(layer)
                (h, c) = internal_state[layer]

                input, c = getattr(self, name)(
                    input, h, c)  # forward propogation call
                internal_state[layer] = (input, c)

            outputs.append([input, c])

        #for i in range(len(outputs)):
        #    print(outputs[i].size())
        return outputs




##############################
#           U-NET
##############################



class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

##############################
#           Generators
##############################


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)


class GeneratorLSTM(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(GeneratorLSTM, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.forward_lstm = CLSTM(
            512, [512], 5, bias=True)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d7_ = torch.unsqueeze(d7, dim=0)
        d8_ = self.forward_lstm(d7_)
        d8 = d8_[0]
        for step in range(1, len(d8_)):
            d8 = torch.cat((d8, d8_[step]), 0)

        u1 = self.up1(d8, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6)

class GeneratorLSTM2(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(GeneratorLSTM2, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)
        #self.forward_lstm = CLSTM(
        #    512, [512], 5, bias=True)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.final_lstm = CLSTM(
            1024, [1024], 5, bias=True)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            #nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        '''
        d7_ = torch.unsqueeze(d7, dim=0)
        d8_ = self.forward_lstm(d7_)
        d8 = d8_[0]
        for step in range(1, len(d8_)):
            d8 = torch.cat((d8, d8_[step]), 0)
        '''
        u1 = self.up1(d7, d6)

        u1_ = torch.unsqueeze(u1, dim=0)
        u1_t_ = self.final_lstm(u1_)
        u1_t = u1_t_[0]
        for step in range(1, len(u1_t_)):
            u1_t = torch.cat((u1_t, u1_t_[step]), 0)
        u2 = self.up2(u1_t, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final(u6)


class GeneratorLSTM_online(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(GeneratorLSTM_online, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.forward_lstm = CLSTM_online(
            512, [512], 5, bias=True)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x, h0, c0):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d7_ = torch.unsqueeze(d7, dim=0)
        d8_ = self.forward_lstm(d7_, h0, c0)
        d8 = d8_[0][0]
        for step in range(1, len(d8_)):
            d8 = torch.cat((d8, d8_[step][0]), 0)

        u1 = self.up1(d8, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)

        return self.final(u6), d8_[-1][0], d8_[-1][1]
##############################
#       Separate LSTM layers
##############################

class CLSTMblock(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=[1],
                 kernel_size=5, bias=True):
        super(CLSTMblock, self).__init__()
        self.forward_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        #layers = [nn.Conv2d(input_channels, input_channels, 5, 1, 2, bias=False)]
        #self.model = nn.Sequential(*layers)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        y = self.forward_net(x)

        #yy = self.forward_net(x)
        #fake_y = yy[0]
        #for step in range(1, len(yy)):
        #    fake_y = torch.cat((fake_y, yy[step]), 0)
        #y = self.model(fake_y)
        return y


##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, in_channels=7):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)



