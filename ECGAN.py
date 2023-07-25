#CGAN Classes: Code for our ECGAN model

from Unet import *

class UNetIICGAN(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=32, num_classes=7):
        super(UNetIICGAN, self).__init__()

        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, input_channels + 1)

        self.upfeature = FeatureMapBlock(input_channels + num_classes, hidden_channels) #Note input dimensions
        self.contract1 = ContractingBlock(hidden_channels, use_in=False, use_dropout=True)
        self.contract2 = ContractingBlock(hidden_channels * 2, use_dropout=True)
        self.contract3 = ContractingBlock(hidden_channels * 4, use_dropout=True)
        self.contract4 = ContractingBlock(hidden_channels * 8)
        self.contract5 = ContractingBlock(hidden_channels * 16)

        self.atrous_conv = AtrousConv(hidden_channels * 32)

        self.expand0 = ExpandingBlock(hidden_channels * 32)
        self.expand1 = ExpandingBlock(hidden_channels * 16)
        self.expand2 = ExpandingBlock(hidden_channels * 8)
        self.expand3 = ExpandingBlock(hidden_channels * 4)
        self.expand4 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

        self.se1 = SE_Block(hidden_channels * 2)
        self.se2 = SE_Block(hidden_channels * 4)
        self.se3 = SE_Block(hidden_channels * 8)

        self.tanh = nn.Tanh()

    def forward(self, x, labels):
        # convert labels to embeddings
        #label_emb = self.label_emb(labels).unsqueeze(-1).unsqueeze(-1)
        #labels = labels.unsqueeze(-1).unsqueeze(-1)
        #print(labels.shape)
        label_emb = self.label_emb(labels).unsqueeze(-1).unsqueeze(-1)
        #print('#')
        #label_emb = self.label_emb(labels).view(x.size(0), self.num_classes, 1, 1)

        #print(label_emb.shape)
        # concatenate the label embeddings with the input tensor
        x = torch.cat((x, label_emb.repeat(1, 1, x.size(2), x.size(3))), dim=1)
        #print(x.shape)

        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x1 = self.se1(x1)
        x2 = self.contract2(x1)
        x2 = self.se2(x2)
        x3 = self.contract3(x2)
        x3 = self.se3(x3)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x5 = self.atrous_conv(x5)
        x6 = self.expand0(x5, x4)
        x7 = self.expand1(x6, x3)
        x8 = self.expand2(x7, x2)
        x9 = self.expand3(x8, x1)
        x10 = self.expand4(x9, x0)
        xn = self.downfeature(x10)

        return self.tanh(xn)

class Discriminator_whole_CGAN(nn.Module):
    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator_whole_CGAN, self).__init__()
        self.label_emb = nn.Embedding(7, 3)
        self.upfeature = FeatureMapBlock(12, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_in=False)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.final = nn.Conv2d(hidden_channels*16, 1, kernel_size=1)

    def forward(self, x, y, labels):
        #gt, input_imgs, labels
        # compute the label embedding
        label_emb = self.label_emb(labels)
        label_emb = label_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.size(2), x.size(3))

        # concatenate the label embedding and input tensors
        x = torch.cat([x, y, label_emb], dim=1)

        # pass the tensor through the network
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn


class Discriminator_mask_CGAN(nn.Module):
    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator_mask_CGAN, self).__init__()
        self.label_emb =nn.Embedding(7, 3)
        self.upfeature = FeatureMapBlock(12, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_in=False)
        self.contract2 = ContractingBlock(hidden_channels*2)
        self.contract3 = ContractingBlock(hidden_channels*4)
        self.contract4 = ContractingBlock(hidden_channels*8)
        self.final = nn.Conv2d(hidden_channels*16, 1, kernel_size=1)
        self.dropout = nn.Dropout()

    def forward(self, x, y, labels):
        # compute the label embedding
        label_emb = self.label_emb(labels)
        label_emb = label_emb.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.size(2), x.size(3))

        # concatenate the label embedding and input tensors
        x = torch.cat([x, y, label_emb], dim=1)

        # pass the tensor through the network
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x2 = self.dropout(x2)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn


def loadm(model, state): #load model pretrained

   model_state_dict = model.state_dict()
   for key in state:
      if  ((key == 'upfeature.conv.weight') ) :
        pass
      else:
        model_state_dict[key] = state[key]

   model.load_state_dict(model_state_dict, strict = False)
   return model

def cgan_inpaint_in(pretrained = True, model_path = "/content/drive/MyDrive/Face_Inpainting_1/Inpaint_UNet.pth"):
    transform = transforms.Compose([
            transforms.ToTensor()
            ])

    input_dim = 6.  #For image+binary map
    output_dim = 3
    disc_dim = 9
    lr = 0.0003
    device = 'cuda'

    gen = UNetIICGAN(input_dim,output_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(),lr=lr)
    disc_whole = Discriminator_whole_CGAN(disc_dim).to(device)
    disc_whole_opt = torch.optim.Adam(disc_whole.parameters(),lr=0.0001)
    disc_mask = Discriminator_mask_CGAN(disc_dim).to(device)
    disc_mask_opt = torch.optim.Adam(disc_mask.parameters(),lr=0.0001)

    if pretrained ==True and model_path != "/content/drive/MyDrive/Face_Inpainting_1/Inpaint_UNet.pth":
        #model_path = "/content/drive/MyDrive/Face_Inpainting_1/Inpaint_UNet.pth"
        loaded_state = torch.load(model_path,map_location=torch.device('cuda'))
        gen.load_state_dict(loaded_state["gen"])
        gen_opt.load_state_dict(loaded_state["gen_opt"])
        disc_whole.load_state_dict(loaded_state["disc_whole"])
        disc_whole_opt.load_state_dict(loaded_state["disc_whole_opt"])
        disc_mask.load_state_dict(loaded_state["disc_mask"])
        disc_mask_opt.load_state_dict(loaded_state["disc_mask_opt"])
    elif pretrained == True and model_path == "/content/drive/MyDrive/Face_Inpainting_1/Inpaint_UNet.pth":
        gen = loadm(gen, loaded_state["gen"])
        disc_whole = loadm(disc_whole, loaded_state["disc_whole"])
        disc_mask = loadm(disc_mask, loaded_state["disc_mask"])


    return gen, gen_opt, disc_whole, disc_whole_opt, disc_mask, disc_mask_opt
