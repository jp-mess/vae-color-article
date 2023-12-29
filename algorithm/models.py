import torch.nn as nn
import torch

# my own KL divergence implementation, forgive me for inefficiency
def KL_loss(pred,label):
  mean_1 = pred[:,:4,:,:]
  var_1 = torch.clip(torch.exp(pred[:,4:,:,:]),min=1e-5)
  mean_2 = label[:,:4,:,:]
  var_2 = torch.clip(torch.exp(label[:,4:,:,:]),min=1e-5)

  kl_loss = lambda mean_1, mean_2, var_1, var_2:(
            torch.log((var_2 / (var_1)) ** 0.5) 
              + (var_1 + (mean_1 - mean_2) ** 2) / (2 * var_2) 
              - 0.5
           )
  loss = kl_loss(mean_1,mean_2,var_1,var_2)
  return loss.mean()

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.ins1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.ins2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.ins1(self.conv1(x)))
        out = self.ins2(self.conv2(out))
        out = out + residual
        return out

class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        pad_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(mode='nearest', scale_factor=upsample)
        pad_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv(out)
        return out





class FCN(nn.Module):
  def __init__(self):
    super().__init__()
    h1 = int(32 * 4) 
    h2 = int(64 * 4) 
    h3 = int(128 * 4)
    self.conv1 = ConvLayer(in_channels=8, out_channels=h1, kernel_size=3, stride=1)
    self.ins1 = nn.InstanceNorm2d(h1, affine=True)
    self.conv2 = ConvLayer(in_channels=h1, out_channels=h2,kernel_size=3,stride=1)
    self.ins2 = nn.InstanceNorm2d(h2, affine=True)
    self.conv3 = ConvLayer(in_channels=h2,out_channels=h3,kernel_size=3,stride=1)
    self.ins3 = nn.InstanceNorm2d(h3, affine=True)

    self.attn1 = SelfAttention(h3,torch.nn.ReLU())
    self.res1 = ResidualBlock(h3)
    self.res2 = ResidualBlock(h3)
    self.res3 = ResidualBlock(h3)
    self.res4 = ResidualBlock(h3)
    self.res5 = ResidualBlock(h3)
    self.attn2 = SelfAttention(h3,torch.nn.ReLU())

    self.deconv1 = UpsampleConvLayer(h3, h2, kernel_size=3, stride=1, upsample=2)
    self.ins4 = nn.InstanceNorm2d(h2, affine=True)
    self.deconv2 = UpsampleConvLayer(h2, h1, kernel_size=3, stride=1, upsample=0.5)
    self.ins5 = nn.InstanceNorm2d(h1, affine=True)
    self.deconv3 = ConvLayer(h1, 8, kernel_size=3, stride=1)

    self.relu = nn.ReLU()

  def forward(self,x):
    out = self.relu(self.ins1(self.conv1(x)))
    out = self.relu(self.ins2(self.conv2(out)))
    out = self.relu(self.ins3(self.conv3(out)))
    out = self.res1(out)
    out = self.res2(out)
    out = self.res3(out)
    out = self.res4(out)
    out = self.res5(out)
    out = self.relu(self.ins4(self.deconv1(out)))
    out = self.relu(self.ins5(self.deconv2(out)))
    out = self.deconv3(out)
    return out
