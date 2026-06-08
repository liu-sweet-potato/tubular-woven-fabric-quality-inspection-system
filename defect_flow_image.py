import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from dev.models import Teacher, Student, AutoEncoder
import cv2
import argparse
from PIL import Image
import time


def parse_args():
    parser = argparse.ArgumentParser(description="参数选择")
    parser.add_argument("--teacher_ckpt", default='assets/weights/ckptSmall/best_teacher.pth', help="")
    parser.add_argument("--model_path", default='assets/weights/ckptSmall')
    #parser.add_argument("--device", default="cuda")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--category", default="zhongyu")
    parser.add_argument("--model_size", default="S")
    parser.add_argument("--channel", default=384)
    parser.add_argument("--ratio", default=0.3)
    parser.add_argument("--resize", default=256)
    parser.add_argument("--score_in_mid_size", default=224)
    args = parser.parse_args()
    return args


def image_crop(image, camera_id, crop_size=530, overlap=0.06):
    """
    将原始图片进行截取,保留有效内容
    0号摄像头 [2800:3330, 2300:5848], (2300, 2800) (5848, 3330)
    1号摄像头 [2700:3230, 2300:5848], (2300, 2700) (5848, 3230)
    2号摄像头 [2600:3130, 2300:5848], (2300, 2600) (5848, 3130)
    3号摄像头 [2750:3280, 2300:5848], (2300, 2750) (5848, 3280)
    """
    x1, y1, x2, y2 = None, None, None, None
    if camera_id == 0:
        x1, y1, x2, y2 = 1800, 2800, 5800, 3330
    elif camera_id == 1:
        x1, y1, x2, y2 = 2300, 2700, 5800, 3230
    elif camera_id == 2:
        x1, y1, x2, y2 = 2300, 2600, 5800, 3130
    elif camera_id == 3:
        x1, y1, x2, y2 = 2300, 2750, 5800, 3280

    eff_image = image.crop((x1, y1, x2, y2))
    image_list = []
    area_list = []
    step_size = int(crop_size * (1 - overlap))
    for i in range(0, eff_image.size[0] - crop_size + 1, step_size):
        crop_img = image.crop((i+x1, y1, i+x1+crop_size, y2))
        image_list.append(crop_img)
        area_list.append([i+x1, y1, i+x1+crop_size, y2])

    return image_list, area_list


def infer_single(args, img):
    # 定位模型
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=torch.device(args.device))
    student_ckpt = torch.load(args.model_path + '/{}_student_last.pth'.format(args.category),
                              map_location=torch.device(args.device))
    ae_ckpt = torch.load(args.model_path + '/{}_autoencoder_last.pth'.format(args.category),
                         map_location=torch.device(args.device))
    teacher = Teacher(args.model_size)
    teacher.load_state_dict(teacher_ckpt)
    student = Student(args.model_size)
    student.load_state_dict(student_ckpt)
    ae = AutoEncoder()
    ae.load_state_dict(ae_ckpt)
    teacher.eval().to(args.device)
    student.eval().to(args.device)
    ae.eval().to(args.device)

    quantiles = np.load(args.model_path + '/{}_quantiles_last.npy'.format(args.category), allow_pickle=True).item()
    qa_st = torch.tensor(quantiles['qa_st'], device=args.device)
    qb_st = torch.tensor(quantiles['qb_st'], device=args.device)
    qa_ae = torch.tensor(quantiles['qa_ae'], device=args.device)
    qb_ae = torch.tensor(quantiles['qb_ae'], device=args.device)
    channel_std = torch.tensor(quantiles['std'], device=args.device)
    channel_mean = torch.tensor(quantiles['mean'], device=args.device)

    with torch.no_grad():
        teacher_output = teacher(img)
        student_output = student(img)
        ae_output = ae(img)

    y_st = student_output[:, :args.channel, :, :]
    y_stae = student_output[:, -args.channel:, :, :]

    normal_teacher_output = (teacher_output - channel_mean) / channel_std

    distance_st = torch.pow(normal_teacher_output - y_st, 2)
    distance_stae = torch.pow(ae_output - y_stae, 2)

    fmap_st = torch.mean(distance_st, dim=1, keepdim=True)
    fmap_st = F.interpolate(fmap_st, size=(256, 256), mode='bilinear')
    fmap_stae = torch.mean(distance_stae, dim=1, keepdim=True)
    fmap_stae = F.interpolate(fmap_stae, size=(256, 256), mode='bilinear')

    normalized_mst = (args.ratio * (fmap_st - qa_st)) / (qb_st - qa_st)
    normalized_mae = (args.ratio * (fmap_stae - qa_ae)) / (qb_ae - qa_ae)
    # combined_map = 0.5 * normalized_mst + 0.5 * normalized_mae
    combined_map = 0.5 * normalized_mst

    return combined_map


def defect_detect(image, camera_id):
    camera_id = int(camera_id)
    start_time = time.time()
    args = parse_args()
    data_transforms = transforms.Compose([transforms.Resize((args.resize, args.resize)), transforms.ToTensor()])
    imgs = Image.fromarray(image)
    imgs, areas = image_crop(imgs, camera_id)

    box_point = []
    for i in range(len(imgs)):
        # img = data_transforms(imgs[i]).to(args.device)
        img = data_transforms(imgs[i]).unsqueeze(0).to(args.device)  # [1, 3, 256, 256]
        combined_map = infer_single(args, img)
        print('time2:', time.time()-start_time)

        # 对 combined_map 的处理
        out_im_np = combined_map[0, 0, :, :].cpu().detach().numpy()
        out_im_np = (out_im_np.clip(0, 1) * 255).astype(np.uint8)

        out_im_thresh1 = cv2.threshold(out_im_np, 127, 255, cv2.THRESH_BINARY)[1]
        out_im_thresh2 = cv2.cvtColor(out_im_thresh1, cv2.COLOR_GRAY2RGB)

        # origin_img = img.cpu().detach().numpy()
        origin_img = img[0].cpu().detach().numpy()
        origin_img = np.transpose(origin_img, (1, 2, 0))
        origin_img_np = (origin_img * 255).astype(np.uint8)
        origin_img_np = cv2.cvtColor(origin_img_np, cv2.COLOR_RGB2BGR)
        # 应用彩色映射
        color_fmap = cv2.applyColorMap(out_im_np, cv2.COLORMAP_JET)
        origin_with_fmap = cv2.addWeighted(origin_img_np, 0.5, color_fmap, 0.5, 0)

        out_im_thresh1 = cv2.resize(out_im_thresh1, (530, 530), interpolation=cv2.INTER_LINEAR)

        '''
        生成检测框
        retval：检测到的连通组件的数量
        label：与输入图像大小相同的数组，其中每个像素的值表示该像素所属的连通组件的标识符
        stats：一个包含每个连通组件统计信息的数组。对于每个组件，统计信息包括（x, y, width, height, area），
              其中 (x, y) 是组件的边界框的左上角坐标，width 和 height 分别是边界框的宽度和高度，area 是组件的面积。
        centroids：每个连通组件的质心坐标数组
        '''
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(out_im_thresh1[30:400, 10:510],
                                                                            connectivity=8)
        stats = stats[stats[:, 4].argsort()]  # 按最后一项area从小到大排序
        bboxs = stats[:-1]  # 去掉图片本身的组件

        for b in bboxs:
            x0, y0 = b[0], b[1]  # 左上角坐标
            x1, y1 = b[0]+b[2], b[1]+b[3]  # 右下角坐标
            # 映射回原图
            iron_start_point = (4320, 2700)
            iron_end_point = (4950, 2850)
            start_point, end_point = (x0+areas[i][0], y0+areas[i][1]), (x1+areas[i][0]+50, y1+areas[i][1]+50)
            # 忽略铁皮的区域
            if camera_id == 1 and (start_point[0] > iron_start_point[0] and start_point[1] > iron_start_point[1]) or \
                    (end_point[0] - 50 < iron_end_point[0] and end_point[1] - 50 < iron_end_point[1]):
                continue
            else:
                box_point.append([start_point, end_point])

    for point in box_point:
        cv2.rectangle(image, point[0], point[1], (255, 0, 0), thickness=3)
    print('time3:', time.time()-start_time)
    if len(box_point) > 0:
        return image, '有异常'
    else:
        return image, '无异常'