import tensorflow as tf
# from utils import vis, load_batch#, load_data
from utils import load_complete_data, show_batch_images
from model import DCGAN, dist_train_step#, train_step
from tqdm import tqdm
import os
import shutil
import pickle
from glob import glob
import wandb
import numpy as np
import cv2
from lstm_kmean.model import TripleNet
import math
import sys

import warnings

def filter_tfa_warning(message, category, filename, lineno, file=None, line=None):
    return not ("tfa_eol_msg.py" in filename and issubclass(category, UserWarning))

warnings.filterwarnings(action="ignore", module=".*tfa_eol_msg.*", category=UserWarning)
warnings.showwarning = filter_tfa_warning

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'

gpus = tf.config.list_physical_devices('GPU')
mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/GPU:1'], 
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
n_gpus = mirrored_strategy.num_replicas_in_sync

latent_dim = 128
input_res  = 128

n_channels  = 14
n_feat      = 128
batch_size  = 128
test_batch_size  = 1
n_classes   = 10

triplenet = TripleNet(n_classes=n_classes)
opt     = tf.keras.optimizers.Adam(learning_rate=3e-4)
triplenet_ckpt    = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='lstm_kmean/experiments/best_ckpt', max_to_keep=5000)
triplenet_ckpt.restore(triplenet_ckptman.latest_checkpoint)
print('TripletNet restored from the latest checkpoint: {}'.format(triplenet_ckpt.step.numpy()))

lr = 3e-4
with mirrored_strategy.scope():
    model        = DCGAN()
    model_gopt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
    model_copt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
    ckpt         = tf.train.Checkpoint(step=tf.Variable(1), model=model, gopt=model_gopt, copt=model_copt)
    ckpt_manager = tf.train.CheckpointManager(ckpt, directory='experiments/best_ckpt', max_to_keep=300)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()


import requests
from  PIL import Image
from transformers import  BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model2 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def image_to_text(path = "C:/Users/devan/Documents/pythonPractice/EEG2Image/data/filtered/apple/0_35.jpg"):
    text = "a photograph of "
    image = Image.open(path)
    inputs = processor(image,text, return_tensors = "pt")

    out = model2.generate(**inputs)
    return (processor.decode(out[0], skip_special_tokens=True))

import os
import random
import colorsys
idx_to_name = {1: "apple", 2: "blue car", 3: "dog", 4: "flower", 5: "gold bracelet", 6: "phone", 7: "scooter", 8: "tiger", 9: "wallet", 10: "watches"}
name_to_idx = {j:i for (i,j) in idx_to_name.items()}
path = "C:/Users/devan/Documents/pythonPractice/EEG2Image/data/filtered"
direc = os.listdir(path)
rel_files = {}
for name in direc:
    true_img_path = path + f'/{name}/true_image.jpg'
    rel_files[name] = list(filter(lambda x: x[-4:] == ".jpg", os.listdir(path + f'/{name}')))
    
def pick_image(idx = 1):
    img_name = random.choice(rel_files[idx_to_name[idx]])[:-4]
    eeg_name = img_name + "_eeg"
    desc_name  = img_name + "_desc" 
    img_path =  path + f"/{idx_to_name[idx]}/" + img_name + ".jpg"
    eeg_path = path + f"/{idx_to_name[idx]}/" + eeg_name + ".npy"
    desc_path = path + f"/{idx_to_name[idx]}/" + desc_name + ".txt"
    
    return img_path, eeg_path, desc_path

import base64
from io import BytesIO

def PIL_to_encode(image):
    
    if image.mode == 'RGBA':
        # Convert the image into RGB
        rgb_image = Image.new("RGB", image.size, (255, 255, 255))  # White background
        rgb_image.paste(image, mask=image.split()[3])  # Paste the image using the alpha channel as a mask
        image = rgb_image
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

def get_images(idx):
    data = {}
    im_path, embed_path, desc_path = pick_image(idx)
    print(im_path, embed_path, desc_path)
    embed_img = visualize_embed(embed_path)
    eeg_image = eeg_vis()
    gen_img = Image.open(im_path)
    text_desc = ""
    with open(desc_path, "r") as f:
        text_desc = f.read()
    data["gen_img"] = PIL_to_encode(gen_img).decode()
    data["eeg_img"] = PIL_to_encode(eeg_image).decode()
    data["embed_img"] = PIL_to_encode(embed_img).decode()
    data["desc"] = text_desc
    return data 

def visualize_embed(file_path = None, eeg_data = None):
    buf = io.BytesIO() 
    # Load the data
    print(file_path)
    if type(eeg_data) == type(None): 
        eeg_data = np.load(file_path)
    # Time vector (assuming a common EEG sampling rate of 256 Hz)
    time = np.linspace(0, len(eeg_data) / 256, len(eeg_data))
    
    # Set font properties for a more formal appearance
    font_properties = {
        'family': 'serif',
        'weight': 'normal',
        'size': 12
    }

    # Plotting the EEG data with updated font properties
    plt.figure(figsize=(10, 5))
    plt.plot(time, eeg_data)
    plt.xlabel('Time (seconds)', fontdict=font_properties)
    plt.ylabel('EEG Amplitude', fontdict=font_properties)
    plt.title('EEG Data Visualization', fontdict={'family': 'serif', 'weight': 'bold', 'size': 14})
    plt.grid(True)
    plt.tight_layout()

    # Update tick labels font
    plt.xticks(fontname='serif')
    plt.yticks(fontname='serif')

    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    return img

# Test the function with the provided EEG data file
# visualize_embed("C:/Users/devan/Documents/pythonPractice/EEG2Image/data/filtered/apple/0_79_eeg.npy")

import matplotlib.pyplot as plt
import io

def eeg_vis(r = -1):
    buf = io.BytesIO() 
    plt.rcParams["font.family"] = "Times New Roman"
    if r == -1:
        r = random.choice(range(45000))
    eeg_data = np.load(f'C:/Users/devan/Documents/pythonPractice/EEG2Image/data/misc_eeg/{r}.npy')
    plt.figure(figsize=(15, 10))
    plt.style.use('seaborn-whitegrid')
    # Shades of green
    green_shades = green_shades = ['#002800','#003C00','#005000','#006400', '#2E8B57', '#228B22', '#556B2F', '#6B8E23', '#3CB371', '#20B2AA', '#32CD32', '#66CDAA', '#9ACD32', '#8FBC8F']
    def darken_color(rgb_color, factor=0.9):
        r, g, b = rgb_color
        h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        l = max(0, min(1, l * factor))
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return r, g, b
    # Convert hex to RGB and then darken the color
    green_shades = [darken_color(tuple(int(col[i:i+2], 16) for i in (1, 3, 5))) for col in green_shades]
    for i in range(14):
        plt.plot(eeg_data[i, :, 0], color=green_shades[i], label=f"Channel {i+1}")
    plt.title("EEG Data for 14 Channels", fontsize=20, fontweight='bold')
    plt.xlabel("Time Points", fontsize=18)
    plt.ylabel("Amplitude", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="upper right", fontsize=12, frameon=True, title="Channels", title_fontsize=14, facecolor='white', edgecolor='black')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    return img

# eeg_vis()

import json
def  pretty_display(dat):
    Image.show(visualize_embed(eeg_data = dat))
    
    X = mirrored_strategy.run(model.gen, args=(tf.expand_dims(dat, axis=0),))
    X = cv2.cvtColor(tf.squeeze(X).numpy(), cv2.COLOR_RGB2BGR)
    X = np.uint8(np.clip((X*0.5 + 0.5)*255.0, 0, 255))

    # Convert the OpenCV image to a PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(X, cv2.COLOR_BGR2RGB))
    Image.show(pil_image)
    
def get_rand_data():
    r  =  random.choice(range(1,134))
    text = ""
    with open(f"C:/Users/devan/Documents/pythonPractice/EEG2Image/data/trans_ex/{r}.txt","r") as f:
        text = f.read()
    dic = json.loads(text)
    return dic["data"]


def generate_story(input_strings):
    # Prepare an advanced prompt for the OpenAI model based on the input strings
    elements = ", ".join(input_strings)
    prompt = (
        f"Create a simple story suitable for text-to-video conversion, including the following elements: {elements}. "
        "The story should have a clear beginning, middle, and end. "
        "Begin with an introduction of the setting and characters, followed by a problem, and then resolve the problem in the end. "
        "Make it short and 3 lines long. Print each of the three lines in a new line. "
    )

    # Use OpenAI to generate a continuation of the prompt
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use "gpt-4" if available, otherwise use "gpt-3.5-turbo"
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )

    story = response['choices'][0]['message']['content'].strip()
    return story


from flask import Flask, request, Response
from  flask_cors import CORS, cross_origin
from functools import wraps
import openai
import requests  

with open('openai_api.json', 'r') as file:
    api_keys = json.load(file)
    openai.api_key = api_keys.get("OPENAI_API_KEY")

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'            

@app.route("/data", methods = ["GET", "POST"])
@cross_origin()
def send_mind_data():
    dic = request.get_json()
    idx = dic["item_idx"]
    if not 1<=idx<=10:
        return Response(status = 489)
    out = get_images(idx)
    return out

def intercepts(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        data  =  request.get_json()
        data["data"] = get_rand_data()
        request._parsed_json_cache = data
        return fn(*args, **kwargs)
    return wrapper

@app.route("/terra_data", methods = ["GET", "POST"])
@cross_origin()
@intercepts
def process_data():
    dic = request.get_json()
    try:
        data = dic["data"]
    except:
        return Response(status = 400)
    
    dat = np.array(data)
    pretty_display(dat)
    
    return Response(status = 200)

@app.route("/raw_data", methods = ["GET", "POST"])
@cross_origin()
def raw_data():
    dic = request.get_json()
    try:
        data = dic["data"]
    except:
        return Response(status = 400)
    
    dat = np.array(data)
    pretty_display(dat)
    
    return Response(status = 200)

@app.route("/text_to_story", methods = ["GET",  "POST"])
def text_to_story():
    dic  = request.get_json()
    arr = dic["data"]
    story = generate_story(arr)
    return story

@app.route("/text_to_image", methods =  ["GET", "POST"])
def text_to_image():
    data = request.json
    response = requests.post('http://127.0.0.1:4000/text_to_images', json = data)
    return response.content

app.run(host='0.0.0.0', port=4000)