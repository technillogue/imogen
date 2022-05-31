#! /bin/bash
set -o xtrace

YOUTUBE_URL="rtmp://a.rtmp.youtube.com/live2"  # URL de base RTMP youtube
KEY="jmt6-73fp-5uu0-wabw-4xg7"                                     # Clé à récupérer sur l'event youtube



#cat $step_dir/* |ffmpeg -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -y -f image2pipe -vcodec png -r 15 -i - -vcodec libx264 -r 15 -pix_fmt yuv420p -crf 17 -preset veryslow



ffmpeg -re \
    -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 \
    -i videos/video.mp4 \
    -f flv \
    -pix_fmt yuvj420p \
    -x264-params keyint=48:min-keyint=48:scenecut=-1 \
    -b:v 4500k \
    -b:a 128k \
    -ar 44100 \
    -acodec aac \
    -vcodec libx264 \
    -preset medium \
    -crf 28 \
    "$YOUTUBE_URL/$KEY"
#    -f lavfi -i anullsrc
#
# Diffusion youtube avec ffmpeg

# Configurer youtube avec une résolution 720p. La vidéo n'est pas scalée.

# VBR="2500k"                                    # Bitrate de la vidéo en sortie
# FPS="30"                                       # FPS de la vidéo en sortie
# QUAL="medium"                                  # Preset de qualité FFMPEG
# YOUTUBE_URL="rtmp://a.rtmp.youtube.com/live2"  # URL de base RTMP youtube

# SOURCE="videos/video.mp4"
# #"udp://239.255.139.0:1234"              # Source UDP (voir les annonces SAP)
# KEY="jmt6-73fp-5uu0-wabw-4xg7"                                     # Clé à récupérer sur l'event youtube

# ffmpeg \
#     -i "$SOURCE" -deinterlace \
#     -vcodec libx264 -pix_fmt yuv420p -preset $QUAL -r $FPS -g $(($FPS * 2)) -b:v $VBR \
#     -acodec libmp3lame -ar 44100 -threads 6 -qscale 3 -b:a 712000 -bufsize 512k \
#     -f flv "$YOUTUBE_URL/$KEY"
