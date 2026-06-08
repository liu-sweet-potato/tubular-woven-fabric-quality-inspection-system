import torch
import torch.nn.functional as F
import os
import torchvision.models as models
from data_process.data_loader import ImageNetDataset, load_infinite
from torch.utils.data import Dataset, DataLoader
from models import Teacher
from torchsummary import summary
import tqdm
from models import wide_resnet101_2
from torchvision import transforms


class DistillationTraining(object):
    def __init__(self, imagenet_dir, channel_size, batch_size, save_path, normalize_iter, iteration=10000, resize=512,
                 model_size='S', wide_resnet_101_arch="Wide_ResNet101_2_Weights.IMAGENET1K_V2", print_freq=10,
                 with_bn=False) -> None:
        self.channel_size = channel_size
        self.mean = torch.empty(channel_size)
        self.std = torch.empty(channel_size)
        self.save_path = save_path
        self.imagenet_dir = imagenet_dir
        self.iteration = iteration
        self.model_size = model_size
        self.batch_size = batch_size
        self.normalize_iter = normalize_iter
        self.wide_resnet_101_arch = wide_resnet_101_arch
        self.print_freq = print_freq
        self.with_bn = with_bn
        self.resize = resize
        self.data_transforms = transforms.Compose([transforms.Resize((resize, resize),),
                                                   transforms.RandomGrayscale(p=0.1),
                                                   transforms.ToTensor()])

    def global_channel_normalize(self, dataloader):
        num = 0
        input_data = torch.randn(1, 3, self.resize, self.resize).cuda()
        temp_tensor = self.pretrain(input_data)
        x = torch.zeros((500, self.channel_size, *temp_tensor.shape[2:]))
        for item in tqdm.tqdm(dataloader):
            if num >= 496:
                break
            ldist = item.cuda()
            y = self.pretrain(ldist).detach().cpu()
            yb = y.shape[0]
            # y.shape = [16, 384, 64, 64]
            x[num:num+yb, :, :, :] = y[:, :, :, :]
            num += yb

        channel_mean = x[:num, :, :, :].mean(dim=(0, 2, 3), keepdim=True).cuda()
        channel_std = x[:num, :, :, :].std(dim=(0, 2, 3), keepdim=True).cuda()
        return channel_mean, channel_std

    def load_pretrain(self):
        self.pretrain = wide_resnet101_2(self.wide_resnet_101_arch, pretrained=True)
        # self.pretrain.load_state_dict(torch.load('pretrained_model.pth'))
        self.pretrain.eval()
        self.pretrain = self.pretrain.cuda()
        # print(summary(self.pretrain, (3, 512, 512)))
    
    def compute_mse_loss(self, teacher, img):
        with torch.no_grad():
            y = self.pretrain(img)  # torch.Size([8, 384, 64, 64])
            y = (y - self.mean)/self.std
        ldistresize = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        y0 = teacher(ldistresize)
        loss = F.mse_loss(y, y0)
        return loss

    def train(self):
        self.load_pretrain()
        imagenet_dataset = ImageNetDataset(self.imagenet_dir, self.data_transforms)
        dataloader = DataLoader(imagenet_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        dataloader = load_infinite(dataloader)
        teacher = Teacher(self.model_size)
        teacher = teacher.cuda()
        # mean_param_path = '{}/imagenet_channel_std.pth'.format(self.save_path)
        # if os.path.exists(mean_param_path):
        #     mean_param = torch.load(mean_param_path)
        #     self.mean = mean_param['mean'].cuda()
        #     self.std = mean_param['std'].cuda()
        # else:
        self.mean, self.std = self.global_channel_normalize(dataloader)
        # torch.save({
        #     'mean': self.mean,
        #     'std': self.std
        # }, '{}/imagenet_channel_std.pth'.format(self.save_path))
        optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * self.iteration), gamma=0.1)
        best_loss = 1000
        loss_accum = 0
        iteration = 0
        print('start train iter:{}'.format(self.iteration))
        for iteration in range(self.iteration):
            # for batch_index, batch_sample in enumerate():
            batch_sample = next(dataloader).cuda()
            teacher.train()
            optimizer.zero_grad()
            loss = self.compute_mse_loss(teacher, batch_sample)
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
            scheduler.step()
            iteration += 1
            if (iteration+1) % self.print_freq == 0 and iteration > 100:
                loss_mean = loss_accum/self.print_freq
                print('iter:{},loss:{:.4f}'.format(iteration, loss_mean))
                if loss_mean < best_loss or best_loss == 1000:
                    best_loss = loss_mean
                    # save teacher
                    print('save best teacher at loss {}'.format(best_loss))
                    teacher.eval()
                    torch.save(teacher.state_dict(), '{}/best_teacher.pth'.format(self.save_path))
                loss_accum = 0

            # save teacher
            teacher.eval()
            torch.save(teacher.state_dict(), '{}/last_teacher.pth'.format(self.save_path))
        

if __name__ == '__main__':
    imagenet_dir = '../assets/datasets/ImageNet'
    channel_size = 384
    save_path = '../assets'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    distillation_training = DistillationTraining(imagenet_dir, channel_size, 16, save_path, normalize_iter=500,
                                                 model_size='S', iteration=10000,
                                                 wide_resnet_101_arch="Wide_ResNet101_2_Weights.IMAGENET1K_V2")
    distillation_training.train()
