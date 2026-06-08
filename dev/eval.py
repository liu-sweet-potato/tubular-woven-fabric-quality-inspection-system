import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import shutil
from torchvision import transforms
from models import Teacher, Student, AutoEncoder
from data_process.data_loader import PipelineDataset
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


class Inference(object):
    def __init__(self, category, val_dir, model_path, ratio=0.1, score_in_mid_size=224, channel=384,
                 result_path='../results/r1', model_size='S', resize=256, device='cuda'):
        self.category = category 
        self.ratio = ratio
        self.resize = resize
        self.score_in_mid_size = score_in_mid_size
        self.val_dir = val_dir
        self.channel = channel
        self.result_path = os.path.join(result_path, label)
        self.teacher = Teacher(model_size)
        self.student = Student(model_size)
        self.ae = AutoEncoder()
        self.model_path = model_path
        self.device = device
        self.load_model()
        self.data_transforms = transforms.Compose([transforms.Resize((resize, resize)),
                                                   transforms.ToTensor()])
        self.gt_transforms = transforms.Compose([transforms.Resize((resize, resize)),
                                                 transforms.ToTensor()])

    def load_model(self,):
        teacher_ckpt = torch.load(self.model_path+'/best_teacher.pth', map_location=torch.device(self.device))
        student_ckpt = torch.load(self.model_path+'/{}_student_last.pth'.format(self.category), map_location=torch.device(self.device))
        ae_ckpt = torch.load(self.model_path+'/{}_autoencoder_last.pth'.format(self.category), map_location=torch.device(self.device))
        self.teacher.load_state_dict(teacher_ckpt)
        self.student.load_state_dict(student_ckpt)
        self.ae.load_state_dict(ae_ckpt)
        self.teacher.eval()
        self.student.eval()
        self.ae.eval()
        self.teacher.to(self.device)
        self.student.to(self.device)
        self.ae.to(self.device)
        quantiles = np.load(self.model_path+'/{}_quantiles_last.npy'.format(self.category), allow_pickle=True).item()
        self.qa_st = torch.tensor(quantiles['qa_st'], device=self.device)
        self.qb_st = torch.tensor(quantiles['qb_st'], device=self.device)
        self.qa_ae = torch.tensor(quantiles['qa_ae'], device=self.device)
        self.qb_ae = torch.tensor(quantiles['qb_ae'], device=self.device)
        self.channel_std = torch.tensor(quantiles['std'], device=self.device)
        self.channel_mean = torch.tensor(quantiles['mean'], device=self.device)

    def infer_single(self, sample_batched):
        img = sample_batched['image']
        img = img.to(self.device)

        with torch.no_grad():
            teacher_output = self.teacher(img)
            student_output = self.student(img)
            ae_output = self.ae(img)

        y_st = student_output[:, :self.channel, :, :]
        y_stae = student_output[:, -self.channel:, :, :]

        normal_teacher_output = (teacher_output-self.channel_mean)/self.channel_std

        distance_st = torch.pow(normal_teacher_output-y_st, 2)
        distance_stae = torch.pow(ae_output-y_stae, 2)

        fmap_st = torch.mean(distance_st, dim=1, keepdim=True)
        fmap_st = F.interpolate(fmap_st, size=(256, 256), mode='bilinear')
        fmap_stae = torch.mean(distance_stae, dim=1, keepdim=True)
        fmap_stae = F.interpolate(fmap_stae, size=(256, 256), mode='bilinear')

        normalized_mst = (self.ratio*(fmap_st-self.qa_st))/(self.qb_st-self.qa_st)
        normalized_mae = (self.ratio*(fmap_stae-self.qa_ae))/(self.qb_ae-self.qa_ae)
        # combined_map = 0.5*normalized_mst + 0.5*normalized_mae
        combined_map = 0.5*normalized_mst
        score_start = (self.resize-self.score_in_mid_size)//2  # 16

        image_score = torch.max(combined_map[:, :, score_start:score_start+self.score_in_mid_size,
                                                   score_start:score_start+self.score_in_mid_size])
        # print('*****', sample_batched['name'], image_score)
        # image_score = torch.max(combined_map)
        return combined_map, image_score

    def eval(self):
        dataset = PipelineDataset(root=self.val_dir,
                                  transform=self.data_transforms,
                                  gt_transform=self.gt_transforms,
                                  phase='test',
                                  category=self.category,
                                  )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        total_pixel_scores = torch.empty(0)
        total_gt_pixel_scores = torch.empty(0)
        scores = []
        gts = []
        
        for i_batch, sample_batched in enumerate(tqdm(dataloader)):
            gts.append(sample_batched['label'].item())
            name = sample_batched['name'][0]
            type = sample_batched['type'][0]
            total_gt_pixel_scores = torch.cat((total_gt_pixel_scores, sample_batched['gt'].view(-1)))
            # 生成预测结果
            combined_map, image_score = self.infer_single(sample_batched)
            scores.append(image_score.item())
            total_pixel_scores = torch.cat((total_pixel_scores, combined_map.detach().cpu().view(-1)))
            out_dir = '{}/{}'.format(self.result_path, type)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_im_path = "{}/{}.bmp".format(out_dir, name)
            # 对 combined_map 的处理
            out_im_np = combined_map[0, 0, :, :].cpu().detach().numpy()
            out_im_np = (out_im_np.clip(0, 1)*255).astype(np.uint8)

            out_im_thresh1 = cv2.threshold(out_im_np, 127, 255, cv2.THRESH_BINARY)[1]  # 转换为二值图像
            out_im_thresh2 = cv2.cvtColor(out_im_thresh1, cv2.COLOR_GRAY2RGB)

            origin_img = sample_batched['image'][0].cpu().detach().numpy()  # (c, h, w)
            origin_img = np.transpose(origin_img, (1, 2, 0))  # (h, w, c)
            origin_img_np = (origin_img * 255).astype(np.uint8)
            origin_img_np = cv2.cvtColor(origin_img_np, cv2.COLOR_RGB2BGR)
            # 应用彩色映射
            color_fmap = cv2.applyColorMap(out_im_np, cv2.COLORMAP_JET)
            origin_with_fmap = cv2.addWeighted(origin_img_np, 0.5, color_fmap, 0.5, 0)

            origin_img_np = cv2.resize(origin_img_np, (550, 550), interpolation=cv2.INTER_LINEAR)
            origin_with_fmap = cv2.resize(origin_with_fmap, (550, 550), interpolation=cv2.INTER_LINEAR)
            out_im_thresh2 = cv2.resize(out_im_thresh2, (550, 550), interpolation=cv2.INTER_LINEAR)
            out_im_thresh1 = cv2.resize(out_im_thresh1, (550, 550), interpolation=cv2.INTER_LINEAR)

            # 生成检测框
            '''
            retval：检测到的连通组件的数量
            label：与输入图像大小相同的数组，其中每个像素的值表示该像素所属的连通组件的标识符
            stats：一个包含每个连通组件统计信息的数组。对于每个组件，统计信息包括（x, y, width, height, area），
                  其中 (x, y) 是组件的边界框的左上角坐标，width 和 height 分别是边界框的宽度和高度，area 是组件的面积。
            centroids：每个连通组件的质心坐标数组
            '''
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(out_im_thresh1, connectivity=8)
            stats = stats[stats[:, 4].argsort()]  # 按最后一项area从小到大排序
            bboxs = stats[:-1]  # 去掉图片本身的组件

            i = 0  # 统计框的个数
            for b in bboxs:
                x0, y0 = b[0], b[1]  # 左上角坐标
                x1 = b[0] + b[2]
                y1 = b[1] + b[3]  # 右下角坐标
                start_point, end_point = (x0, y0), (x1, y1)
                # 在原图上画框
                # cv2.rectangle(origin_img_np, start_point, end_point, (0, 0, 255), thickness=2)
                # 在识别图上画框
                cv2.rectangle(out_im_thresh2, start_point, end_point, (0, 0, 255), thickness=1)
                i += 1

            # 创建水平堆叠的图像
            out_hstack = np.hstack((origin_img_np, origin_with_fmap, out_im_thresh2))
            # 保存结果图像
            cv2.imwrite(out_im_path,out_hstack)
            cv2.imshow('img', out_hstack)
            cv2.waitKey(1)
        print('********************************true*************************************')
        gtnp = np.array(gts)
        scorenp = np.array(scores)
        total_gt_pixel_scoresnp = total_gt_pixel_scores.cpu().detach().numpy().astype('uint8')
        total_pixel_scoresnp = total_pixel_scores.cpu().detach().numpy()

        # auroc = roc_auc_score(gtnp,scorenp)
        if total_gt_pixel_scoresnp.max()==0:
            # print("label:{},auroc:{:.4f}".format(self.category,auroc))
            return
        # auroc_pixel = roc_auc_score(total_gt_pixel_scoresnp,total_pixel_scoresnp)
        ap_pixel = average_precision_score(total_gt_pixel_scoresnp,total_pixel_scoresnp)
        ap = average_precision_score(gtnp,scorenp)
        # print("label:{},auroc:{:.4f},auroc_pixel:{:.4f},ap:{:.4f},ap_pixel:{:.4f}".format(self.category,auroc,auroc_pixel,ap,ap_pixel))
        print("label:{},ap:{:.4f},ap_pixel:{:.4f}".format(self.category, ap, ap_pixel))


if __name__ == "__main__":
    val_dir = '../assets/datasets/'
    model_path = '../assets/weights/ckptSmall'
    label = "zhongyu"
    # ckptSmall1, ratio=0.2
    # ckptSmall2, ratio=0.3
    # infer = Inference(label, val_dir, model_path, ratio=0.2, model_size='S', device='cuda')
    infer = Inference(label, val_dir, model_path, ratio=0.2, model_size='S', device='cpu')
    infer.eval()
