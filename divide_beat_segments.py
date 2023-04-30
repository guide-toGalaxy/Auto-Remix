import librosa
import os
import sys
import soundfile as sf
from glob import glob


def divide_to_beats(aud):
	# Load audio file
	audio_file = aud
	aud=aud.split("/")[2:][0]
	y, sr = librosa.load(audio_file)

	# Get the tempo and beats
	tempo, beat_samples = librosa.beat.beat_track(y=y, sr=sr,units="samples")
	start_sample=0
	# Write the beat components to new audio files
	for i, beat_start in enumerate(beat_samples):
	    #print("BEAT FRAMES",i,len(beat_frames),)
	    beat_end = beat_start
	    beat = y[start_sample:beat_end] 
	    start_sample=beat_end
	    output_file = os.path.join(output_dir, aud[:-4]+f'_{i+1}.wav')
	    sf.write(output_file, beat, samplerate=sr,subtype="PCM_24")
	# last_sample=y[start_sample:]
	# sf.write(output_dir+aud[:-4]+f'_beat{len(beat_samples)+1}.wav', last_sample, samplerate=sr,subtype="PCM_24")


if __name__ == "__main__":
	input_dir = sys.argv[1]
	songs_to_remix=glob("./"+input_dir+"/*.mp3")

	output_dir = sys.argv[2]
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for aud in songs_to_remix:
		divide_to_beats(aud)




