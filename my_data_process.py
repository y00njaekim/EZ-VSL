import argparse
import sys
from moviepy.editor import VideoFileClip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm


def get_arguments():
    parser = argparse.ArgumentParser(description="Extract clips and audio from videos.")
    parser.add_argument("--train", action="store_true", help="Process for training data")
    parser.add_argument("--test", action="store_true", help="Process for test data")
    
    return parser.parse_args(), parser


def extract_clips_and_audio(filename, path, clip_duration=3):
    clip = VideoFileClip(filename)
    total_duration = int(clip.duration)
    
    if not os.path.exists(os.path.join(path, "audio")):
        os.makedirs(os.path.join(path, "audio"))
    if not os.path.exists(os.path.join(path, "frames")):
        os.makedirs(os.path.join(path, "frames"))
    
    progress_bar = tqdm(total=total_duration, desc="Processing", unit="s")
    
    for start_time in range(0, total_duration, clip_duration):
        end_time = min(start_time + clip_duration, total_duration)
        if end_time == total_duration:
            break
        try:
            subclip = clip.subclip(start_time, end_time)
            
            frame = subclip.get_frame(clip_duration / 2)
            
            frame_filename = f"{os.path.basename(filename)}_frame_{start_time}.jpg"
            frame_path = os.path.join(path, "frames", frame_filename)
            
            Image.fromarray(np.uint8(frame)).save(frame_path)
    
            audio_filename = f"{os.path.basename(filename)}_audio_{start_time}.wav"
            audio_path = os.path.join(path, "audio", audio_filename)
            
            subclip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        
        progress_bar.update(clip_duration)
    
    progress_bar.close()


def main(args, parser):
    if args.train:
        path = "./dataset/train"
    elif args.test:
        path = "./dataset/test"
    else:
        parser.print_help()
        sys.exit(1)
    
    for idx, file in enumerate(os.listdir(os.path.join(path, "raw"))):
        print(f"Processing file {idx + 1}/{len(os.listdir(os.path.join(path, 'raw')))}: {file}...")
        extract_clips_and_audio(os.path.join(path, "raw", file), path)
    
    
if __name__ == "__main__":
    main(*get_arguments())