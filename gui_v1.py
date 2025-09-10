# import gradio as gr
# import torch
# from basicsr.archs.edsr_arch import EDSR
# from basicsr.utils import tensor2img
# import cv2

# # Load your model
# model = EDSR(num_in_ch=3,
#             num_out_ch=3,
#             num_feat=192,
#             num_block=6,
#             upscale=4,
#             num_heads=6,
#             hw=8,
#             ww=8,
#             img_range=255.,
#             rgb_mean=[0.4488, 0.4371, 0.4040])
# model.load_state_dict(torch.load('experiments/EDSR_Lx4_f192b6_DF2K_500k_B8G1_001/models/net_g_latest.pth')['params'])
# model.eval()

# def inference(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0) / 255.
#     with torch.no_grad():
#         out = model(img_tensor)
#     out_img = tensor2img(out)
#     return out_img

# gr.Interface(fn=inference,
#              inputs=gr.Image(type="numpy"),
#              outputs="image",
#              title="BasicSR Model Tester").launch()


import gradio as gr
import torch
from basicsr.archs.edsr_arch import EDSR
from basicsr.utils import tensor2img
import cv2

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your model
model = EDSR(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=192,
    num_block=6,
    upscale=4,
    num_heads=6,
    hw=8,
    ww=8,
    img_range=255.,
    rgb_mean=[0.4488, 0.4371, 0.4040]
).to(device)  # send model to GPU

checkpoint = torch.load(
    'experiments/EDSR_Lx4_f192b6_DF2K_500k_B8G1_001/models/net_g_latest.pth',
    map_location=device
)
model.load_state_dict(checkpoint['params'])
model.eval()

def inference(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(device) / 255.  # send input to GPU
    with torch.no_grad():
        out = model(img_tensor)
    out_img = tensor2img(out.detach().cpu())  # move back to CPU for numpy conversion
    return out_img

# import os

# def inference(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_tensor = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(device) / 255.  # send input to GPU
#     with torch.no_grad():
#         out = model(img_tensor)
#     out_img = tensor2img(out.detach().cpu())  # move back to CPU for numpy conversion

#     # Save the image in JPG format
#     save_path = "output.jpg"
#     cv2.imwrite(save_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
#     print(f"Image saved at {os.path.abspath(save_path)}")

#     return out_img

gr.Interface(
    fn=inference,
    inputs=gr.Image(type="numpy"),
    outputs="image",
    title="BasicSR Model Tester (GPU)"
).launch()
