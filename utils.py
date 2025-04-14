import time
import os 
from IPython.display import clear_output, display
from contextlib import nullcontext
import paddle
from PIL import Image
def empty_cache():
    import gc
    gc.collect()
    paddle.device.cuda.empty_cache()

def check_is_model_complete(path = './NovelAI_latest_ab21ba3c_paddle', check_vae_size=300000000):
    return os.path.exists(os.path.join(path,'vae/model_state.pdparams')) and\
         os.path.getsize(os.path.join(path,'vae/model_state.pdparams')) > check_vae_size

def save_image_info(image, path = './outputs/'):
    os.makedirs(path, exist_ok=True)
    cur_time = time.time()
    seed = image.argument['seed']
    filename = f'{cur_time}_SEED_{seed}'
    image.save(os.path.join(path, filename + '.png'), quality=100)
    with open(os.path.join(path, filename + '.txt'), 'w') as f:
        for key, value in image.argument.items():
            f.write(f'{key}: {value}\n')
    
def ReadImage(image, height = None, width = None):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    # clever auto inference of image size
    w, h = image.size
    if height == -1 or width == -1:
        if w > h:
            width = 768
            height = max(64, round(width / w * h / 64) * 64)
        else: # w < h
            height = 768
            width = max(64, round(height / h * w / 64) * 64)
        if width > 576 and height > 576:
            width = 576
            height = 576
    if (height is not None) and (width is not None) and (w != width or h != height):
        image = image.resize((width, height), Image.ANTIALIAS)
    return image

class StableDiffusionFriendlyPipeline():
    def __init__(self, superres_pipeline = None):
        self.pipe = None
        self.pipe_i2i = None
        self.model = "./NovelAI_latest_ab21ba3c_paddle"
        if not check_is_model_complete(self.model):
            if os.path.exists(self.model):
                print(self.model + '解压不完全! 请重启内核, 删除该文件夹后重新解压!')
            self.model = "./model_pruned_paddle"
        self.vae = 'animevae.pdparams'
        self.remote_vae = 'data/data171442/animevae.pdparams'

        self.superres_pipeline = superres_pipeline

    def from_pretrained(self, verbose = True, force = False):
        model = self.model
        vae = self.vae
        if (not force) and self.pipe is not None:
            return

        if verbose: print('!!!!!正在加载模型, 请耐心等待, 如果出现两行红字是正常的, 不要惊慌!!!!!')
        from diffusers_paddle import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

        # text to image
        self.pipe = StableDiffusionPipeline.from_pretrained(model)
        
        if vae is not None:
            print('正在换用 vae......')
            local_vae = os.path.join(os.path.join(self.model, 'vae'), self.vae)
            if (not os.path.exists(local_vae)) or os.path.getsize(local_vae) < 300000000:
                print('初次使用, 正在复制 vae...... (等 %s/vae/animevae.pdparams 文件约 319MB 即可)'%self.model)
                from shutil import copy
                copy(self.remote_vae, local_vae) # copy from remote, avoid download everytime

            self.pipe.vae.load_state_dict(paddle.load(local_vae)) # 换用更好的 vae (有效果!)

        # image to image
        pipe = self.pipe
        self.pipe_i2i = StableDiffusionImg2ImgPipeline(vae=pipe.vae,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,
        unet=pipe.unet,scheduler=pipe.scheduler,safety_checker=pipe.safety_checker,feature_extractor=pipe.feature_extractor)

        # save space on GPU as we do not need them (safety check has been turned off)
        del self.pipe.safety_checker
        del self.pipe_i2i.safety_checker
        self.pipe.safety_checker = None
        self.pipe_i2i.safety_checker = None

        if verbose: print('成功加载完毕')

    def run(self, opt, task = 'txt2img'):
        self.from_pretrained()
        seed = None if opt.seed == -1 else opt.seed
        # precision_scope = paddle.amp.auto_cast if opt.precision=="autocast" else nullcontext
        # PRECISION = "fp16" if opt.precision=="autocast" else "fp32"

        task_func = None
        if task == 'txt2img':
            def task_func():
                return self.pipe(opt.prompt, seed=seed, width=opt.width, height=opt.height, guidence_scale=opt.guidence_scale, 
                                num_inference_steps=opt.num_inference_steps, negative_prompt=opt.negative_prompt).images[0]
        elif task == 'img2img':
            def task_func():
                return self.pipe_i2i(opt.prompt, seed=seed, 
                                init_image=ReadImage(opt.image_path, height=opt.height, width=opt.width), 
                                num_inference_steps=opt.num_inference_steps, 
                                strength=opt.strength, guidance_scale=opt.guidence_scale, negative_prompt=opt.negative_prompt)[0][0]
        
        with nullcontext():
            for i in range(opt.num_return_images):
                empty_cache()
                image = task_func()
                
                # super resolution
                if (self.superres_pipeline is not None):
                    argument = image.argument
                    argument['superres_model_name'] = opt.superres_model_name
                    
                    image = self.superres_pipeline.run(opt, image = image, end_to_end = False)
                    image.argument = argument

                save_image_info(image, opt.output_dir)
                if i % 5 == 0:
                    clear_output()

                display(image)
                print('Seed =', image.argument['seed'])

class SuperResolutionPipeline():
    def __init__(self):
        self.model = None
        self.model_name = ''
    
    def run(self, opt, 
                image = None, 
                task = 'superres', 
                end_to_end = True,
                force_empty_cache = True
            ):
        """
        end_to_end: return PIL image if False, display in the notebook and autosave otherwise
        empty_cache: force clear the GPU cache by deleting the model
        """
        if opt.superres_model_name is None or opt.superres_model_name in ('','无'):
            return image

        import numpy as np
        if image is None:
            image = ReadImage(opt.image_path, height=None, width=None) # avoid resizing
        image = np.array(image)
        image = image[:,:,[2,1,0]]  # RGB -> BGR

        empty_cache()
        if self.model_name != opt.superres_model_name:
            if self.model is not None:
                del self.model 

            import logging
            logging.disable(100)    
            # [ WARNING] - The _initialize method in HubModule will soon be deprecated, you can use the __init__() to handle the initialization of the object
            import paddlehub as hub
            # print('正在加载超分模型! 如果出现两三行红字是正常的, 不要担心哦!')
            self.model = hub.Module(name = opt.superres_model_name)
            logging.disable(30)
            
        self.model_name = opt.superres_model_name

        # time.sleep(.1) # wait until the warning prints
        # print('正在超分......请耐心等待')
    
        try:
            image = self.model.reconstruct([image], use_gpu = (paddle.device.get_device() != 'cpu'))[0]['data']
        except:
            print('图片尺寸过大, 超分时超过显存限制')
            self.empty_cache(force_empty_cache)
            paddle.disable_static()
            return

        image = image[:,:,[2,1,0]] # BGR -> RGB
        image = Image.fromarray(image)
        
        self.empty_cache(force_empty_cache)
        paddle.disable_static()

        if end_to_end:
            cur_time = time.time()
            image.save(os.path.join(opt.output_dir,f'Highres_{cur_time}.png'),quality=100)
            clear_output()
            display(image)
            return
        return image
    
    def empty_cache(self, force = True):
        # NOTE: it seems that ordinary method cannot clear the cache
        # so we have to delete the model (?)
        if not force:
            return
        del self.model
        self.model = None
        self.model_name = ''


class StableDiffusionSafetyCheckerEmpty(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
