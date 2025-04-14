# Code credits to 凉心半浅良心人
# Has modified
#似乎有点问题

import os
from IPython.display import clear_output
from utils import StableDiffusionFriendlyPipeline, SuperResolutionPipeline
# from stable_diffusion import StableDiffusionTask, empty_cache
from tqdm.auto import tqdm
import paddle

_ENABLE_ENHANCE = False

if paddle.device.get_device() != 'cpu':
    # settings for super-resolution, currently not supporting multi-gpus
    # see docs at https://github.com/PaddlePaddle/PaddleHub/tree/develop/modules/image/Image_editing/super_resolution/falsr_a
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

pipeline_superres = SuperResolutionPipeline()
pipeline = StableDiffusionFriendlyPipeline(superres_pipeline = pipeline_superres)
    
####################################################################
#
#                  Image Enhancement (Abandoned)
#
####################################################################
if _ENABLE_ENHANCE:
    from enhance_utils import RealX

    gfp = None
    from pathlib import Path
    def enhance_batch(realx, image_dir, output_dir="big_repaired", do_repaire=False, do_upscale=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        newpath = [p for p in Path(image_dir).ls() if "png" in str(p)]
        for path in tqdm(newpath):
            name = path.name
            p = str(path)
            repaired = None
            if do_repaire:
                try:
                    repaired = gfp.enhance(p)
                except:
                    repaired = None

            if not do_upscale:
                if repaired is not None:
                    repaired.save(output_dir / name)
            else:
                if repaired is None:
                    big_image = realx.enhance(p)
                else:
                    big_image = realx.enhance(repaired)
                big_image.save(output_dir / name)


    upscale_model_name  = ""
    def enhance_run(opt):
        global upscale_model_name
        global realx
        if opt.upscale_model_name != upscale_model_name:
            realx = RealX(opt.upscale_model_name)
            upscale_model_name = opt.upscale_model_name
        empty_cache()
        run_enhance(opt, realx)
        
    def run_enhance(opt, realx):
        if not os.path.exists(opt.image_dir):
            print("输入的文件夹地址不存在，请检查 {opt.image_dir}")
            return
        enhance_batch(realx, image_dir=opt.image_dir, output_dir=opt.upscale_image_dir)
        print(f"放大后的图片已经保存至{opt.upscale_image_dir}文件夹，请查看！")



####################################################################
#
#                     Graphics User Interface
#
####################################################################
# Code to turn kwargs into Jupyter widgets
import ipywidgets as widgets
from collections import OrderedDict


# Allows long widget descriptions
style  = {'description_width': 'initial'}
# Force widget width to max
layout = widgets.Layout(width='100%')

def get_widget_extractor(widget_dict):
    # allows accessing after setting, this is to reduce the diff against the argparse code
    class WidgetDict(OrderedDict):
        def __getattr__(self, val):
            return self[val].value
    return WidgetDict(widget_dict)


class StableDiffusionUI():
    def __init__(self, pipeline = pipeline):
        self.widget_opt = OrderedDict()
        self.pipeline = pipeline
        self.gui = None
        self.run_button = None
        self.run_button_out = widgets.Output()
        self.task = 'txt2img'

    def on_run_button_click(self, b):
        with self.run_button_out:
            clear_output()
            self.pipeline.run(get_widget_extractor(self.widget_opt), task = self.task)
    

class StableDiffusionUI_txt2img(StableDiffusionUI):
    def __init__(self):
        super().__init__()
        widget_opt = self.widget_opt
        widget_opt['prompt'] = widgets.Textarea(
            layout=layout, style=style,
            description='prompt描述' + '&nbsp;' * 22,
            value="extremely detailed CG unity 8k wallpaper,black long hair,cute face,1 adult girl,happy, green skirt dress, flower pattern in dress,solo,green gown,art of light novel,in field",
            disabled=False
        )
        widget_opt['negative_prompt'] = widgets.Textarea(
            layout=layout, style=style,
            description='negative_prompt反面描述 <br />',
            value="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
            disabled=False
        )

        widget_opt['height'] = widgets.IntText(
            layout=layout, style=style,
            description='图片的高度(像素), 64的倍数',
            value=512,
            disabled=False
        )

        widget_opt['width'] = widgets.IntText(
            layout=layout, style=style,
            description='图片的宽度(像素), 64的倍数',
            value=512,
            disabled=False
        )

        widget_opt['num_return_images'] = widgets.IntText(
            layout=layout, style=style,
            description='生成图片数量' + '&nbsp;'*22,
            value=1,
            disabled=False
        )

        widget_opt['num_inference_steps'] = widgets.IntText(
            layout=layout, style=style,
            description='推理的步数' + '&nbsp;'*25,
            value=50,
            disabled=False
        )

        widget_opt['guidence_scale'] = widgets.FloatText(
            layout=layout, style=style,
            description='cfg' + '&nbsp;'*37,
            value=7.5,
            disabled=False
        )

        widget_opt['output_dir'] = widgets.Text(
            layout=layout, style=style,
            description='图片的保存路径' + '&nbsp;'*18,
            value="outputs",
            disabled=False
        )
        # widget_opt['precision'] = widgets.Dropdown(
        #     layout=layout, style=style,
        #     description='evaluate at this precision',
        #     value="full",
        #     options=["full", "autocast"],
        #     disabled=False
        # )
        widget_opt['seed'] = widgets.IntText(
            layout=layout, style=style,
            description='随机数种子(-1表示不设置随机种子)',
            value=-1,
            disabled=False
        )

        widget_opt['superres_model_name'] = widgets.Dropdown(
            layout=layout, style=style,
            description='超分模型的名字' + '&nbsp;'*18,
            value="falsr_a",
            options=["falsr_a", "falsr_b", "falsr_c", "无"],
            disabled=False
        )

        self.run_button = widgets.Button(
            description='点击生成图片！',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to run (settings will update automatically)',
            icon='check'
        )
        
        self.run_button.on_click(self.on_run_button_click)
        
        if _ENABLE_ENHANCE:
            widget_opt['image_dir'] = widgets.Text(
                layout=layout, style=style,
                description='需要放大图片的文件夹地址。',
                value="outputs",
                disabled=False
            )
            widget_opt['upscale_image_dir'] = widgets.Text(
                layout=layout, style=style,
                description='放大后的图片所要保存的文件夹地址。',
                value="upscale_outputs",
                disabled=False
            )
            widget_opt['upscale_model_name'] = widgets.Dropdown(
                layout=layout, style=style,
                description='放大图片所用的模型',
                value="RealESRGAN_x4plus",
                options=[
                        'RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B', 'RealSR',
                        'RealESRGAN_x4', 'RealESRGAN_x2', 'RealESRGAN_x8'
                    ],
                disabled=False
            )

            enhance_button = widgets.Button(
                description='开始放大图片！',
                disabled=False,
                button_style='', # 'success', 'info', 'warning', 'danger' or ''
                tooltip='Click to run (settings will update automatically)',
                icon='check'
            )
            enhance_button_out = widgets.Output()
            def on_enhance_button_click(b):
                with run_button_out:
                    clear_output()
                with enhance_button_out:
                    clear_output()
                    enhance_run(get_widget_extractor(widget_opt))
            enhance_button.on_click(on_enhance_button_click)
    
        self.gui = widgets.VBox([widget_opt[k] for k in list(widget_opt.keys())]
                            +   [self.run_button, self.run_button_out])
    



class StableDiffusionUI_img2img(StableDiffusionUI):
    def __init__(self):
        super().__init__()

        widget_opt = self.widget_opt
        widget_opt['prompt'] = widgets.Textarea(
            layout=layout, style=style,
            description='prompt描述' + '&nbsp;' * 22,
            value="Kurisu Makise, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress",
            disabled=False
        )
        widget_opt['negative_prompt'] = widgets.Textarea(
            layout=layout, style=style,
            description='negative_prompt反面描述 <br />',
            value="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
            disabled=False
        )

        widget_opt['image_path'] = widgets.Text(
            layout=layout, style=style,
            description='需要转换的图片路径' + '&nbsp;'*12,
            value='image_Kurisu.png',
            disabled=False
        )

        widget_opt['height'] = widgets.IntText(
            layout=layout, style=style,
            description='图片的高度, -1为自动判断' + '&nbsp;'*2,
            value=-1,
            disabled=False
        )
        
        widget_opt['width'] = widgets.IntText(
            layout=layout, style=style,
            description='图片的宽度, -1为自动判断' + '&nbsp;'*2,
            value=-1,
            disabled=False
        )

        widget_opt['num_return_images'] = widgets.IntText(
            layout=layout, style=style,
            description='生成图片数量' + '&nbsp;'*22,
            value=1,
            disabled=False
        )

        widget_opt['num_inference_steps'] = widgets.IntText(
            layout=layout, style=style,
            description='推理的步数' + '&nbsp;'*25,
            value=50,
            disabled=False
        )

        widget_opt['strength'] = widgets.FloatText(
            layout=layout, style=style,
            description='修改强度' + '&nbsp;'*29,
            value=0.8,
            disabled=False
        )

        widget_opt['guidence_scale'] = widgets.FloatText(
            layout=layout, style=style,
            description='cfg' + '&nbsp;'*37,
            value=7.5,
            disabled=False
        )

        widget_opt['output_dir'] = widgets.Text(
            layout=layout, style=style,
            description='图片的保存路径' + '&nbsp;'*18,
            value="outputs",
            disabled=False
        )
        # widget_opt['precision'] = widgets.Dropdown(
        #     layout=layout, style=style,
        #     description='evaluate at this precision',
        #     value="full",
        #     options=["full", "autocast"],
        #     disabled=False
        # )
        
        widget_opt['superres_model_name'] = widgets.Dropdown(
            layout=layout, style=style,
            description='超分模型的名字' + '&nbsp;'*19,
            value="无",
            options=["falsr_a", "falsr_b", "falsr_c", "无"],
            disabled=False
        )

        widget_opt['seed'] = widgets.IntText(
            layout=layout, style=style,
            description='随机数种子(-1表示不设置随机种子)',
            value=-1,
            disabled=False
        )

        self.run_button = widgets.Button(
            description='点击生成图片！',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to run (settings will update automatically)',
            icon='check'
        )
        self.run_button.on_click(self.on_run_button_click)

        self.gui = widgets.VBox([widget_opt[k] for k in list(widget_opt.keys())]
                            +   [self.run_button, self.run_button_out])

        self.task = 'img2img'


class SuperResolutionUI(StableDiffusionUI):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        widget_opt = self.widget_opt
        
        widget_opt['image_path'] = widgets.Text(
            layout=layout, style=style,
            description='需要超分的图片路径' ,
            value='image_Kurisu.png',
            disabled=False
        )

        widget_opt['superres_model_name'] = widgets.Dropdown(
            layout=layout, style=style,
            description='超分模型的名字' + '&nbsp;'*6,
            value="falsr_a",
            options=["falsr_a", "falsr_b", "falsr_c"],
            disabled=False
        )

        widget_opt['output_dir'] = widgets.Text(
            layout=layout, style=style,
            description='图片的保存路径' + '&nbsp;'*6,
            value="outputs",
            disabled=False
        )
        
        self.run_button = widgets.Button(
            description='点击超分图片！',
            disabled=False,
            button_style='success', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click to run (settings will update automatically)',
            icon='check'
        )
        self.run_button.on_click(self.on_run_button_click)

        self.gui = widgets.VBox([widget_opt[k] for k in list(widget_opt.keys())]
                            +   [self.run_button, self.run_button_out])

        self.task = 'superres'

# instantialization
gui_txt2img = StableDiffusionUI_txt2img()
gui_img2img = StableDiffusionUI_img2img()
gui_superres = SuperResolutionUI(pipeline = pipeline_superres)
