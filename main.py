#@title 开始!

# 模型和权重加载都在这里,pipe 加载的权重,vae_path 加载的本体,如果第一次运行下面报错, 就重启内核重新运行一次
from diffusers_paddle import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import paddle
from utils import save_image_info
import os
pipe = StableDiffusionPipeline.from_pretrained("./CompVis-stable-diffusion-v1-4") # 加载SD默认权重
#pipe = StableDiffusionPipeline.from_pretrained("./waifu-diffusion") # 加载waifu权重
#pipe = StableDiffusionPipeline.from_pretrained("./NovelAI_latest_ab21ba3c_paddle") # 加载NovelAI权重
# pipe = StableDiffusionPipeline.from_pretrained("./model_pruned_paddle")

vae_path = 'stable-diffusion-v1-4/model_state.pdparams'  # 加载 SD默认  模型
#vae_path = 'waifu-diffusion/unet/model_state.pdparams'   # 加载 waifu  模型
#vae_path = './NovelAI_latest_ab21ba3c_paddle/vae/animevae.pdparams' # 加载 NovelAI vae 模型
pipe.vae.load_state_dict(paddle.load(vae_path)) # 换用更好的 vae (有效果!)

# 图生图
pipe_i2i = StableDiffusionImg2ImgPipeline(vae=pipe.vae,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,
unet=pipe.unet,scheduler=pipe.scheduler,safety_checker=pipe.safety_checker,feature_extractor=pipe.feature_extractor)
print('加载完毕')
