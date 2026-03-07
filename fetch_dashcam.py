import urllib.request
import os

def download_dashcam():
    url = "https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_videos/solidWhiteRight.mp4"
    save_path = "data/sample_dashcam.mp4"
    
    print(f"Downloading TRUE FPP dashcam video from {url}...")
    try:
        # Unconditionally overwrite
        if os.path.exists(save_path):
            os.remove(save_path)
            
        urllib.request.urlretrieve(url, save_path)
        print(f"Successfully downloaded to: {save_path}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    download_dashcam()
