# coding: utf-8
#import structured_as_functions_attempt

# get_ipython().system('nvidia-smi')


def install_dependencies():
    # @title Instalación de bibliotecas
    # @markdown Esta celda tardará un poco porque tiene que descargar varias librerías
     
    print("Descargando CLIP...")
    # !git clone https://github.com/openai/CLIP                 
    # print("Instalando bibliotecas de Python para IA...")
    # !git clone https://github.com/CompVis/taming-transformers 
    # !pip install ftfy regex tqdm omegaconf pytorch-lightning torch 
    # !pip install kornia                                       
    # !pip install einops                                       
     
    # print("Instalando bibliotecas para manejo de metadatos...")
    # !pip install stegano                                      
    # !apt install exempi                                       
    # !pip install python-xmp-toolkit                           
    # !pip install imgtag                                       
    # !pip install pillow==7.1.2                                
     
    # print("Instalando bibliotecas de Python para creación de vídeos...")
    # !pip install imageio-ffmpeg 
    # !mkdir steps
    # print("Instalación finalizada.")
    # !pip install torchvision
    # ! pip install imageio


# In[ ]:

def download_model():
    #@title Selección de modelos a descargar
    #@markdown Por defecto, el notebook descarga el modelo 16384 de ImageNet. Existen otros como ImageNet 1024, COCO-Stuff, WikiArt 1024, WikiArt 16384, FacesHQ o S-FLCKR, que no se descargan por defecto, ya que sería en vano si no los vas a usar, así que si quieres usarlos, simplemente selecciona los modelos a descargar.

    imagenet_1024 = False #@param {type:"boolean"}
    imagenet_16384 = True #@param {type:"boolean"}
    coco = False #@param {type:"boolean"}
    faceshq = False #@param {type:"boolean"}
    wikiart_1024 = False #@param {type:"boolean"}
    wikiart_16384 = False #@param {type:"boolean"}
    sflckr = False #@param {type:"boolean"}

    # if imagenet_1024:
    #   !curl -L -o vqgan_imagenet_f16_1024.yaml -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.yaml' #ImageNet 1024
    #   !curl -L -o vqgan_imagenet_f16_1024.ckpt -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.ckpt'  #ImageNet 1024
    # if imagenet_16384:
    #   !curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.yaml' #ImageNet 16384
    #   !curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.ckpt' #ImageNet 16384
    # if coco:
    #   !curl -L -o coco.yaml -C - 'https://dl.nmkd.de/ai/clip/coco/coco.yaml' #COCO
    #   !curl -L -o coco.ckpt -C - 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt' #COCO
    # if faceshq:
    #   !curl -L -o faceshq.yaml -C - 'https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT' #FacesHQ
    #   !curl -L -o faceshq.ckpt -C - 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt' #FacesHQ
    # if wikiart_1024: 
    #   !curl -L -o wikiart_1024.yaml -C - 'http://mirror.io.community/blob/vqgan/wikiart.yaml' #WikiArt 1024
    #   !curl -L -o wikiart_1024.ckpt -C - 'http://mirror.io.community/blob/vqgan/wikiart.ckpt' #WikiArt 1024
    # if wikiart_16384: 
    #   !curl -L -o wikiart_16384.yaml -C - 'http://mirror.io.community/blob/vqgan/wikiart_16384.yaml' #WikiArt 16384
    #   !curl -L -o wikiart_16384.ckpt -C - 'http://mirror.io.community/blob/vqgan/wikiart_16384.ckpt' #WikiArt 16384
    # if sflckr:
    #   !curl -L -o sflckr.yaml -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1' #S-FLCKR
    #   !curl -L -o sflckr.ckpt -C - 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1' #S-FLCKR



