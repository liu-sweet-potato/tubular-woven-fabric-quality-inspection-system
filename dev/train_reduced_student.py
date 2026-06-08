import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import numpy as np
import tqdm
import argparse
import yaml
import os
from sklearn.metrics import roc_auc_score, average_precision_score
from loguru import logger
from models import Teacher, Student, AutoEncoder
from data_process.data_loader import load_infinite, PipelineDataset, ImageNetDataset


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/mvtec_train.yaml')
    parser.add_argument('--category', type=str, default='')
    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--iterations', type=int, default=None)
    args = parser.parse_args()
    return args


def bool_constructor(loader, node):
    value = loader.construct_scalar(node)
    return value.lower() == 'true'


yaml.add_constructor('tag:yaml.org,2002:bool', bool_constructor, yaml.SafeLoader)


class Reduced_Student_Teacher(object):
    def __init__(self, config):
        self.config = config
        self.category = config['category']
        self.ckpt_dir = config['Model']['ckpt_dir']
        model_size = config['Model']['model_size']
        with_bn = config['Model'].get('with_bn', False)
        with_bn = str(with_bn).lower() == 'true'
        self.device = config['Model']['device']
        self.quantile_tresh = config['Model']['quantile_tresh']
        self.channel_size = config['Model']['channel_size']
        self.student = Student(model_size, with_bn).cuda(self.device)
        # self.student.apply(weights_init)
        self.teacher = Teacher(model_size, with_bn)
        self.load_pretrain_teacher()
        self.ae = AutoEncoder(is_bn=with_bn).cuda(self.device)
        # self.ae.apply(weights_init)
        resize = config['Model']['input_size']
        self.score_in_mid_size = int(0.9 * resize)
        self.resize = resize
        self.fmap_size = (resize, resize)
        self.channel_mean, self.channel_std = None, None
        self.batch_size = config['Model']['batch_size']
        self.print_freq = config['print_freq']
        self.data_transforms = transforms.Compose([transforms.Resize((resize, resize)),
                                                   transforms.ToTensor()])
        self.gt_transforms = transforms.Compose([transforms.Resize((resize, resize)),
                                                 transforms.ToTensor()])
        teacher_input = config['Datasets']['imagenet']['teacher_input']
        grayscale_ratio = config['Datasets']['imagenet']['grayscale_ratio']
        self.data_transforms_imagenet = transforms.Compose([transforms.Resize((teacher_input, teacher_input)),
                                                            transforms.RandomGrayscale(p=grayscale_ratio),
                                                            transforms.CenterCrop((resize, resize)),
                                                            transforms.ToTensor()])
        self.set_seed(config['seed'])

    """为三个不同的随机数生成器设置相同的种子值（seed）。为了确保在程序或实验中获得可重复的结果。"""
    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def load_pretrain_teacher(self):
        self.teacher.load_state_dict(torch.load(self.ckpt_dir + '/best_teacher.pth'))
        self.teacher = self.teacher.cuda(self.device)
        self.teacher.eval()
        for parameters in self.teacher.parameters():
            parameters.requires_grad = False
        logger.info('load teacher model from {}'.format(self.ckpt_dir + '/best_teacher.pth'))

    def global_channel_normalize(self, dataloader):
        num = 0
        input_data = torch.randn(1, 3, self.resize, self.resize).cuda(self.device)
        temp_tensor = self.teacher(input_data)
        x = torch.zeros((500, self.channel_size, *temp_tensor.shape[2:]))
        for item in tqdm.tqdm(dataloader):
            if num >= 500:
                break
            ldist = item['image'].cuda(self.device)
            y = self.teacher(ldist).detach().cpu()
            yb = y.shape[0]

            x[num:num + yb, :, :, :] = y[:, :, :, :]
            num += yb
        self.channel_mean = x[:num, :, :, :].mean(dim=(0, 2, 3), keepdim=True).cuda(self.device)
        self.channel_std = x[:num, :, :, :].std(dim=(0, 2, 3), keepdim=True).cuda(self.device)
        return self.channel_mean, self.channel_std
    '''
    def teacher_normalization(teacher, train_loader):

        mean_outputs = []
        for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
            if on_gpu:
                train_image = train_image.cuda()
            teacher_output = teacher(train_image)
            mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
            mean_outputs.append(mean_output)
        channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
        channel_mean = channel_mean[None, :, None, None]

        mean_distances = []
        for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
            if on_gpu:
                train_image = train_image.cuda()
            teacher_output = teacher(train_image)
            distance = (teacher_output - channel_mean) ** 2
            mean_distance = torch.mean(distance, dim=[0, 2, 3])
            mean_distances.append(mean_distance)
        channel_var = torch.mean(torch.stack(mean_distances), dim=0)
        channel_var = channel_var[None, :, None, None]
        channel_std = torch.sqrt(channel_var)

        return channel_mean, channel_std
    '''
    def loss_st(self, image, imagenet_iterator, teacher: Teacher, student: Student):
        with torch.no_grad():
            t_pdn_out = teacher(image)
            normal_t_out = (t_pdn_out - self.channel_mean) / self.channel_std
        s_pdn_out = student(image)
        s_pdn_out = s_pdn_out[:, :self.channel_size, :, :]
        distance_s_t = torch.pow(normal_t_out - s_pdn_out, 2)
        dhard = torch.quantile(distance_s_t[:8, :, :, :], self.quantile_tresh)
        hard_data = distance_s_t[distance_s_t >= dhard]
        Lhard = torch.mean(hard_data)
        image_p = next(imagenet_iterator)
        s_imagenet_out = student(image_p[0].cuda(self.device))
        N = torch.mean(torch.pow(s_imagenet_out[:, :self.channel_size, :, :], 2))
        loss_st = Lhard + N
        return loss_st

    def loss_ae(self, image, teacher: Teacher, student: Student, autoencoder: AutoEncoder):
        aug_img = image.cuda(self.device)
        with torch.no_grad():
            t_out = teacher(aug_img)
            normal_t_out = (t_out - self.channel_mean) / self.channel_std
        ae_out = autoencoder(aug_img)
        s_pdn_out = student(aug_img)
        s_pdn_out = s_pdn_out[:, self.channel_size:, :, :]
        distance_ae = torch.pow(normal_t_out - ae_out, 2)
        distance_stae = torch.pow(ae_out - s_pdn_out, 2)
        LAE = torch.mean(distance_ae)
        LSTAE = torch.mean(distance_stae)
        return LAE, LSTAE

    def caculate_channel_std(self, dataloader):
        channel_std_ckpt = "{}/{}_good_dataset_channel_std.pth".format(self.ckpt_dir, self.category)
        self.channel_mean, self.channel_std = self.global_channel_normalize(dataloader)
        logger.info('channel mean:{}'.format(self.channel_mean.shape), 'channel std:{}'.format(self.channel_std.shape))
        channel_std = {'mean': self.channel_mean,
                       'std': self.channel_std}
        torch.save(channel_std, channel_std_ckpt)

    def load_datasets(self):
        # normalize_dataset
        normalize_dataset = PipelineDataset(root=self.config['Datasets']['train']['root'],
                                            transform=self.data_transforms,
                                            gt_transform=self.gt_transforms,
                                            phase='train',
                                            category=self.category,
                                            split_ratio=1)
        normalize_dataloader = DataLoader(normalize_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        # dataset
        dataset = PipelineDataset(root=self.config['Datasets']['train']['root'],
                                  transform=self.data_transforms,
                                  gt_transform=self.gt_transforms,
                                  phase='train',
                                  category=self.category,
                                  split_ratio=0.8)
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        train_dataloader = load_infinite(train_dataloader)
        logger.info('load train dataset:length:{}'.format(len(dataset)))
        # quantile_dataset
        quantile_dataset = PipelineDataset(root=self.config['Datasets']['train']['root'],
                                           transform=self.data_transforms,
                                           gt_transform=self.gt_transforms,
                                           phase='eval',
                                           category=self.category,
                                           split_ratio=0.8)
        quantile_dataloader = DataLoader(quantile_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        # imagenet
        imagenet = ImageNetDataset(root=self.config['Datasets']['imagenet']['root'],
                                   transform=self.data_transforms_imagenet)
        imagenet_loader = DataLoader(imagenet, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
        imagenet_iterator = load_infinite(imagenet_loader)
        # eval_dataloader
        eval_dataloader = ""
        return normalize_dataloader, train_dataloader, imagenet_iterator, quantile_dataloader, eval_dataloader

    def train(self, iterations=70000):
        normalize_dataloader, train_dataloader, imagenet_iterator, quantile_dataloader, \
                                                                                eval_dataloader = self.load_datasets()
        self.caculate_channel_std(normalize_dataloader)
        optimizer = torch.optim.Adam(list(self.student.parameters()) + list(self.ae.parameters()), lr=0.0001,
                                     weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * iterations), gamma=0.1)

        best_loss = 100
        logger.info('start train iter:', iterations)
        for i_batch in range(iterations):
            sample_batched = next(train_dataloader)
            image = sample_batched['image'].cuda(self.device)
            self.student.train()
            self.ae.train()
            loss_st = self.loss_st(image, imagenet_iterator, self.teacher, self.student)
            LAE, LSTAE = self.loss_ae(image, self.teacher, self.student, self.ae)
            loss_total = loss_st + LAE + LSTAE

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            if i_batch % self.print_freq == 0:
                logger.info("label:{},batch:{}/{},loss_total:{:.4f},loss_st:{:.4f},loss_ae:{:.4f},loss_stae:{:.4f}".
                             format(self.category, i_batch, iterations, loss_total.item(), loss_st.item(), LAE.item(),
                                    LSTAE.item()))
                self.qa_st, self.qb_st, self.qa_ae, self.qb_ae = self.map_norm_quantiles(quantile_dataloader)

                if loss_total < best_loss:
                    best_loss = loss_total
                    print('saving model in {}'.format(self.ckpt_dir))
                    torch.save(self.student.state_dict(),'{}/{}_student.pth'.format(self.ckpt_dir,self.category))
                    torch.save(self.ae.state_dict(),'{}/{}_autoencoder.pth'.format(self.ckpt_dir,self.category))
                    quantiles = {
                        'qa_st':self.qa_st,
                        'qb_st':self.qb_st,
                        'qa_ae':self.qa_ae,
                        'qb_ae':self.qb_ae,
                        'std':self.channel_std.cpu().numpy(),
                        'mean':self.channel_mean.cpu().numpy()
                    }
                    np.save('{}/{}_quantiles.npy'.format(self.ckpt_dir,self.category),quantiles)
                torch.save(self.student.state_dict(), '{}/{}_student_last.pth'.format(self.ckpt_dir, self.category))
                torch.save(self.ae.state_dict(), '{}/{}_autoencoder_last.pth'.format(self.ckpt_dir, self.category))
                quantiles = {
                    'qa_st': self.qa_st,
                    'qb_st': self.qb_st,
                    'qa_ae': self.qa_ae,
                    'qb_ae': self.qb_ae,
                    'std': self.channel_std.cpu().numpy(),
                    'mean': self.channel_mean.cpu().numpy()
                }
                np.save('{}/{}_quantiles_last.npy'.format(self.ckpt_dir, self.category), quantiles)

    def eval(self, eval_dataloader):
        scores = []
        gts = []
        for sample_batched in tqdm.tqdm(eval_dataloader):
            gts.append(sample_batched['label'].item())
            combined_map, image_score = self.infer_single(sample_batched)
            scores.append(image_score.item())
        gtnp = np.array(gts)
        scorenp = np.array(scores)
        auroc = roc_auc_score(gtnp, scorenp)
        return auroc

    def infer_single(self, sample_batched):
        img = sample_batched['image']
        img = img.cuda(self.device)
        with torch.no_grad():
            teacher_output = self.teacher(img)
            student_output = self.student(img)
            ae_output = self.ae(img)

        y_st = student_output[:, :self.channel_size, :, :]
        y_stae = student_output[:, -self.channel_size:, :, :]

        normal_teacher_output = (teacher_output - self.channel_mean) / self.channel_std

        distance_st = torch.pow(normal_teacher_output - y_st, 2)
        distance_stae = torch.pow(ae_output - y_stae, 2)

        fmap_st = torch.mean(distance_st, dim=1, keepdim=True)
        fmap_stae = torch.mean(distance_stae, dim=1, keepdim=True)
        fmap_st = F.interpolate(fmap_st, size=(self.resize, self.resize), mode='bilinear')
        fmap_stae = F.interpolate(fmap_stae, size=(self.resize, self.resize), mode='bilinear')
        normalized_mst = (0.1 * (fmap_st - self.qa_st)) / (self.qb_st - self.qa_st)
        normalized_mae = (0.1 * (fmap_stae - self.qa_ae)) / (self.qb_ae - self.qa_ae)
        combined_map = 0.5 * normalized_mst + 0.5 * normalized_mae
        score_start = (self.resize - self.score_in_mid_size) // 2
        image_score = torch.max(combined_map[:, :, score_start:score_start + self.score_in_mid_size,
                                                   score_start:score_start + self.score_in_mid_size])
        return combined_map, image_score

    def map_norm_quantiles(self, dataloader):
        xst, xae = [], []
        self.student.eval()
        self.ae.eval()
        self.teacher.eval()
        for i_batch, sample_batched in enumerate(dataloader):
            sample_batched = sample_batched['image'].cuda(self.device)
            with torch.no_grad():
                t_out = self.teacher(sample_batched)
                s_out = self.student(sample_batched)
                ae_out = self.ae(sample_batched)

            y_st = s_out[:, :self.channel_size, :, :]
            y_stae = s_out[:, -self.channel_size:, :, :]
            # normal_t_out = self.compute_normalize_teacher_out(t_out)
            normal_t_out = (t_out - self.channel_mean) / self.channel_std
            distance_s_t = torch.pow(normal_t_out - y_st, 2)
            distance_stae = torch.pow(ae_out - y_stae, 2)
            anomaly_map_st_by_c = torch.mean(distance_s_t, dim=1)
            anomaly_map_stae_by_c = torch.mean(distance_stae, dim=1)
            anomaly_map_st = F.interpolate(anomaly_map_st_by_c.unsqueeze(0), size=self.fmap_size, mode='bilinear')
            anomaly_map_ae = F.interpolate(anomaly_map_stae_by_c.unsqueeze(0), size=self.fmap_size, mode='bilinear')
            xst.append(anomaly_map_st.detach().cpu().numpy())
            xae.append(anomaly_map_ae.detach().cpu().numpy())
        qa_st = np.percentile(np.concatenate(xst), 90)
        qb_st = np.percentile(np.concatenate(xst), 99.5)
        qa_ae = np.percentile(np.concatenate(xae), 90)
        qb_ae = np.percentile(np.concatenate(xae), 99.5)
        return qa_st, qb_st, qa_ae, qb_ae


if __name__ == '__main__':
    args = get_arguments()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    rst = Reduced_Student_Teacher(config=config)
    rst.train(iterations=config['Model']['iterations'])
