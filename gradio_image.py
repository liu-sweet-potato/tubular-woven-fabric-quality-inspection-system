import gradio as gr
from defect_flow_image import defect_detect


# 创建Gradio界面
# -------------------图片模式输入组件-------------------
input_img = gr.Image(label="摄像头画面", type="numpy")
input_id = gr.Number(label="摄像头ID", value=0, precision=0)
input_list = [input_img, input_id]
# -------------------图片模式输出组件-------------------
output_img = gr.Image(label="识别结果")
output_label = gr.Label(label="带坯异常检测结果")
output_list = [output_img, output_label]

# 标题
title = "“四象明眸”——基于EfficientAD的圆织机经纬线异常视觉检测系统"

# 描述
description = "<div align='center'>在带坯生产过程中，会出现圆织机经纬线不良品，使用蒸馏模型自动化检测不良品以提升检测效率。"

iface = gr.Interface(
    fn=defect_detect,
    inputs=input_list,
    outputs=output_list,
    title=title,
    description=description,
    #allow_flagging="never",
    flagging_mode="never",
)

# 启动Gradio界面
iface.launch(server_port=9999)