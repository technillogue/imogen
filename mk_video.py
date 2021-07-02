def video(i):
    init_frame = 1 #Este es el frame donde el vídeo empezará
    last_frame = i #Puedes cambiar i a el número del último frame que quieres generar. It will raise an error if that number of frames does not exist.

    min_fps = 10
    max_fps = 30

    total_frames = last_frame-init_frame

    length = 15 #Tiempo deseado del vídeo en segundos

    frames = []
    tqdm.write('Generando video...')
    for i in range(init_frame,last_frame): #
        filename = f"steps/{i:04}.png"
        frames.append(Image.open(filename))

    #fps = last_frame/10
    fps = np.clip(total_frames/length,min_fps,max_fps)

    from subprocess import Popen, PIPE
    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264', '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', 'video.mp4'], stdin=PIPE)
    for im in tqdm(frames):
        im.save(p.stdin, 'PNG')
    p.stdin.close()

    print("El vídeo está siendo ahora comprimido, espera...")
    p.wait()
    print("El vídeo está listo")

