from utils import jsonlload, parse_args,jsondump
from googletrans import Translator

import copy

from ACD_models import ACD_model

from transformers import AutoTokenizer
import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os
import natsort
from stqdm import stqdm

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

task_name = 'ABSA' 

label_id_to_name = ['True', 'False']

opt = parse_args()

st.markdown("<h2 style='text-align: center; color: black;'>🎥 졸업작품 테스트 🎥</h2>",
            unsafe_allow_html=True)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    opt = parser.parse_args()
    return opt


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def main(opt):
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    elif opt.dpm:
        sampler = DPMSolverSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(opt.repeat)]
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad(), \
        precision_scope("cuda"), \
        model.ema_scope():
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples, _ = sampler.sample(S=opt.steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)

                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        st.image(image=img)
                        base_count += 1
                        sample_count += 1

                    all_samples.append(x_samples)

            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            grid = put_watermark(grid, wm_encoder)
            grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

def predict_from_korean_form(text):
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained('paust/pko-t5-base')
    GCU_T5_1_Path = "/home/nlplab/hdd1/졸업작품/류상연/saved_model_epoch_5.pt"

    GCU_T5_1 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_1.load_state_dict(torch.load(GCU_T5_1_Path, map_location=device))
    GCU_T5_1.to(device)
    GCU_T5_1.eval()
    sentences = "summarize: " + text 

    tokenized_data = tokenizer(sentences, padding='max_length', max_length=1024, truncation=True)

    input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
    attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
        #생성 모델 생성
    outputs = GCU_T5_1.model_PLM.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=5, max_length=512, early_stopping=True)
        
    pred_t = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
    test = pred_t[0]
    print(test)
    style = test.split(',',maxsplit=1)
    docs = style[1].split('.')
    st.write('Summarized text')
    st.write(test)
    translator = Translator()
    for i in range(len(docs)-1):
        docs[i] = translator.translate(docs[i], dest="en").text
        docs[i] = docs[i] + ',4k,detailed,' + style[0]
        # docs[i] = docs[i].remove('.')
    docs.pop()

    return docs

def make_image_conti(text):
    # 이미지를 경로에서 불러와서 6개씩 묶어서 페이지 만들기
        image_name = natsort.natsorted(os.listdir(image_path))
        image_list = []
        # st.write("# 이미지 리스트를 사용하여 이미지 접근")
        for i in range(len(image_name)):
            img_name = image_path + "/" + image_name[i]
            image_list.append(Image.open(img_name))
            # st.image(image_list[i])

        st.write("# RESULT")
        rest_image = len(image_list)
        for i in range(0, len(image_list), 5): # 이부분 수정 필요함
            background = Image.open("externalFile.jpg")
            draw = ImageDraw.Draw(background)
            if rest_image > 5:
                for j in range(0, 5):
                    img_name = image_path + "/" + image_name[i*6+j]
                    im = Image.open(img_name)
                    y = (100 + j*165)
                    im = im.resize((270, 150))
                    background.paste(im = im, box = (140, y))
                    draw.text((100, y), f"{j+1}", (0, 0, 0), font = ImageFont.truetype("Ubuntu-R.ttf", 20))
                    text_range = int(len(text[i*6+j])/27)
                    if text_range > 0:
                        for k in range(text_range+1):
                            draw.text((440, y+10+15*k), text[i*6+j][26*k:26*k+26], (0, 0, 0), font = ImageFont.truetype("Ubuntu-R.ttf", 15))
                    else:
                        draw.text((440, y+10), text[i*6+j], (0, 0, 0), font = ImageFont.truetype("Ubuntu-R.ttf", 15))
                        draw = ImageDraw.Draw(background)
                rest_image = rest_image-5
            else:
                for j in range(rest_image):
                    img_name = image_path + "/" + image_name[i*5+j]
                    im = Image.open(img_name)
                    y = (100 + j*165)
                    im = im.resize((270, 150))
                    background.paste(im = im, box = (140, y))
                    draw.text((100, y), f"{j+1}", (0, 0, 0), font = ImageFont.truetype("Ubuntu-R.ttf", 20))
                    text_range = int(len(text[i*6+j])/27)
                    if text_range > 0:
                        for k in range(text_range+1):
                            draw.text((440, y+10+15*k), text[i*6+j][26*k:26*k+26], (0, 0, 0), font = ImageFont.truetype("Ubuntu-R.ttf", 15))
                    else:
                        draw.text((440, y+10), text[i*6+j], (0, 0, 0), font = ImageFont.truetype("Ubuntu-R.ttf", 15))
                    draw = ImageDraw.Draw(background)
            st.image(background, f"{i+1} page")
            background.save("result.jpg")

tab1, tab2 = st.tabs(
    ["Image generation", "Summarize"])

image_path = "/home/nlplab/hdd1/졸업작품/류상연/scripts/outputs/txt2img-samples/samples"

with tab1:
    input = st.text_input("Docs")
    button = st.button("실행하기")
    if button:
        st.write("input text")
        st.write(input)
        # 이전 이미지 삭제
        if os.path.exists(image_path):
            for file in os.scandir(image_path):
                os.remove(file.path)
        text = predict_from_korean_form(input)
        st.write(text)
        # 이미지 생성
        opt = parse_args()
        opt.ckpt = '../v2-1_512-ema-pruned.ckpt'
        opt.config = '../configs/stable-diffusion/v2-inference.yaml'
        for i in range(len(text)):
            st.write(text[i])
            with st.spinner('Wait for it...'):
                opt.prompt = text[i]
                main(opt)
        make_image_conti(text)



if __name__ == "__main__":
    args = parse_args()
