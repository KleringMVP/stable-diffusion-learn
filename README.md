# stable-diffusion-learn

## 一、进入 Notebook 并启动 GPU 环境
打开项目后，点击 “运行一下”。

选择 GPU 环境（每天赠送 8 小时）。

进入项目后等待加载完成。

## 二、首次使用：解压模型 & 安装依赖
只在第一次使用时运行以下步骤！

解压模型文件。

安装必要的依赖包。

更新 pip 和安装 PaddleNLP 等。

```python
import os
from IPython.display import clear_output
from utils import check_is_model_complete

print('正在解压模型')
if not check_is_model_complete("./NovelAI_latest_ab21ba3c_paddle"):
    !unzip -o "/home/aistudio/data/data171442/NovelAI_latest_ab21ba3c_paddle.zip"

if not os.path.exists("diffusers_paddle"):
    !unzip -o diffusers_paddle.zip

print('正在安装库')
!pip install --upgrade pip -i https://mirror.baidu.com/pypi/simple
!pip install -U fastcore paddlenlp ftfy regex --user -i https://mirror.baidu.com/pypi/simple
!pip install --upgrade paddlenlp -i https://mirror.baidu.com/pypi/simple

clear_output()
print('加载完毕, 下次不用再运行这里了')

## 三、加载 Stable Diffusion 模型
之后每次运行项目都从这一步开始：
```python
from diffusers_paddle import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import paddle
import os

pipe = StableDiffusionPipeline.from_pretrained("./CompVis-stable-diffusion-v1-4")

vae_path = 'stable-diffusion-v1-4/model_state.pdparams'
pipe.vae.load_state_dict(paddle.load(vae_path))

pipe_i2i = StableDiffusionImg2ImgPipeline(
    vae=pipe.vae,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    unet=pipe.unet,
    scheduler=pipe.scheduler,
    safety_checker=pipe.safety_checker,
    feature_extractor=pipe.feature_extractor
)

print('模型加载完毕')

##四、文本生成图像
```python
height = 512
width = 768
seed = 3354985548
steps = 50
cfg = 7

prompt = "miku, looking at viewer, long hair, standing, 1girl, hair ornament, cute, jacket, white flower, white dress"

image = pipe(prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=cfg, seed=seed).images[0]

##五、批量生成图像
```python
number = 3
height = 768
width = 512
steps = 50
cfg = 17.5
prompt = "kitsune made out of flames, digital art, synthwave"
negative_prompt = "lowres, bad anatomy, bad hands, blurry"

for i in range(number):
    image = pipe(prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=cfg, negative_prompt=negative_prompt).images[0]

##六、自训练模型使用
```python
learned_embeded_path = "sd-concepts-library/your-trained-style.pdparams"
for token, embeds in paddle.load(learned_embeded_path).items():
    pipe.tokenizer.add_tokens(token)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    with paddle.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[token_id] = embeds

prompt = f"a colorful color painting of {token} styles, kitsune made out of flames, digital art"
images = pipe([prompt], height=704, width=512, num_inference_steps=50, guidance_scale=7.5).images
