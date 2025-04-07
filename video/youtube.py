
import yt_dlp
import os

def download(dir_path, save_name, url):
    outtmpl = os.path.join(dir_path, f"{save_name}.%(ext)s")
    ydl_opts = {
        'outtmpl': outtmpl,
        'format': 'bestvideo[height<=1040][ext=mp4]',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    video_path = "C:/Users/alex6/Desktop/video"
    save_name = '더들리'
    url = 'https://www.youtube.com/shorts/H2sIZoWk31s'
