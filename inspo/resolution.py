# via https://colab.research.google.com/github/justinjohn0306/VQGAN-CLIP/blob/main/VQGAN%2BCLIP_(Zooming)_(z%2Bquantize_method_with_addons).ipynb?fbclid=IwAR03gy5ZM0hn80ZFMO7Ywc6OumsJ0fMZLjKPgC7BGHJjm0KQ-IxOQxZN3nc#scrollTo=lp0_TdGHHjtM


#@markdown ## **Install SRCNN** 

#git clone https://github.com/Mirwaisse/SRCNN.git
#curl https://raw.githubusercontent.com/justinjohn0306/SRCNN/master/models/model_2x.pth -o model_2x.pth
#curl https://raw.githubusercontent.com/justinjohn0306/SRCNN/master/models/model_3x.pth -o model_3x.pth
#curl https://raw.githubusercontent.com/justinjohn0306/SRCNN/master/models/model_4x.pth -o model_4x.pth#@markdown ## **Increase Resolution**

# import subprocess in case this cell is run without the above cells
import subprocess
# Set zoomed = True if this cell is run
zoomed = True

init_frame = 1#@param {type:"number"}
last_frame = i#@param {type:"number"}

for i in range(init_frame, last_frame): #
    filename = f"{i:04}.png"
    cmd = [
        'python',
        '/content/SRCNN/run.py',
        '--zoom_factor',
        '2',  # Note if you increase this, you also need to change the model.
        '--model',
        '/content/model_2x.pth',  # 2x, 3x and 4x are available from the repo above
        '--image',
        filename,
        '--cuda'
    ]
    print(f'Upscaling frame {i}')

    process = subprocess.Popen(cmd, cwd=f'{working_dir}/steps/')
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        print(
            "You may be able to avoid this error by backing up the frames,"
            "restarting the notebook, and running only the video synthesis cells,"
            "or by decreasing the resolution of the image generation steps. "
            "If you restart the notebook, you will have to define the `filepath` manually"
            "by adding `filepath = 'PATH_TO_THE_VIDEO'` to the beginning of this cell. "
            "If these steps do not work, please post the traceback in the github."
        )
        raise RuntimeError(stderr)
