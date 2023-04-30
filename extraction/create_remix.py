import numpy as np
import pickle
import sys
import wave
import math
import random
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
random.seed(15) 
from glob import glob
import pathlib
from pathlib import Path


def output_file(input_files):
	output_file = "combined_files.wav"

# Open the first input file to get the audio parameters
	with wave.open(input_files[0], "rb") as audio_file:
		audio_params = audio_file.getparams()

    # Open the output file for writing
		with wave.open(output_file, "wb") as output:
        # Set the output file parameters to match the input file parameters
			output.setparams(audio_params)

        # Iterate through the input files and write their data to the output file
			for input_file in input_files:
				with wave.open(input_file, "rb") as audio_file:
					output.writeframes(audio_file.readframes(audio_params.nframes))
	



def get_features_from_path(path,mapping,features):
	array_index=mapping.index(path)
	return features[array_index]
	
def get_neighbours(sample,num,model,features,mapping):
	sample = sample[np.newaxis, ...]
	distances, array_indexes = model.kneighbors(sample,n_neighbors=num)
	paths=index_to_path(array_indexes[0],mapping)
	return paths,distances[0]

def index_to_path(indexes,mapping):
	paths=[mapping[x] for x in indexes]
	return paths
	
def get_track_name(file_path):
	track=file_path.split('/')
	#print(track)
	track=("".join(track[-1]))
	#print(track)
	track_name_1=track.split('_')
	track_name=("".join(track_name_1[0]))
	return (track_name)

def get_track_number(file_path):
	track=file_path.split('/')
	#print(track)
	track=("".join(track[-1]))
	#print(track)
	track_name_1=track.split('_')
	track_name=("".join(track_name_1[-1]))
	
	#print(track_name[:-4])
	return int(track_name[:-4])


def get_track_from_name_and_number(file_path,number):
	track=file_path.split('_')
	#print(track)
	track[-1]=str(number)+".wav"
	#print("_".join(track))
	return ("_".join(track))
	#track[-1].split

def change_track(last_track_beats,surprisal_factor,x):
	threshold=0
	if last_track_beats > 0:
		threshold=surprisal_factor*(math.log(last_track_beats,10))
		random_num=random.random()
		threshold_values[x]=threshold
		random_no[x]=random_num
	if  random_num<= threshold:
		return True
	return False
	
def get_beat_from_diff_track(neighbours):
	current_track=get_track_name(neighbours[0])
	for i in neighbours[1:]:
		this_track=get_track_name(i)
		if(this_track!=current_track and i not in used):
			return i

def get_random():
	wav_files = glob("../my_beats_2/*.wav")
	return (random.choice(wav_files))

no_of_samples_in_remix=int(sys.argv[1])
surprisal_factor = float(sys.argv[2])
print(surprisal_factor)
threshold_values=np.zeros(no_of_samples_in_remix)
random_no=np.zeros(no_of_samples_in_remix)

with open('mapping.pkl', 'rb') as f:
	mapping = pickle.load(f)
with open('dataset.pkl','rb') as f:
	features=pickle.load(f)
with open('nn.pkl','rb') as f:
	NN=pickle.load(f)
used=[]

def generate(no_of_samples,surprisal_factor,mapping,features,NN):
	t=0
	appender=[]
	#last_beat_path=np.random.choice(mapping)
	#print(last_beat_path)
	last_beat_path="../my_beats_2/Don't Stay_53.wav"
	used.append(last_beat_path)
	appender.append(last_beat_path)
	last_track_name=get_track_name(last_beat_path)
	last_track_number=get_track_number(last_beat_path)
	same_track_beats=1
	
	current_feature=get_features_from_path(last_beat_path,mapping,features)
	
	#print("FIRST BEAT",first_beat_path)
	nearest_paths,distances=(get_neighbours(current_feature,500,NN,features,mapping))
	#get_beat_from_diff_track(nearest_paths)
	
	for j in range(no_of_samples_in_remix):
		if(change_track(same_track_beats,surprisal_factor,j)):
			last_beat_path=get_beat_from_diff_track(nearest_paths)
			#last_beat_path=get_random()
			used.append(last_beat_path)
			#print("YO",last_beat_path)
			last_track_name=get_track_name(last_beat_path)
			last_track_number=get_track_number(last_beat_path)
			appender.append(last_beat_path)
			#print(t,last_beat_path)
			t=t+1
			
			current_feature=get_features_from_path(last_beat_path,mapping,features)
			
			nearest_paths,distances=(get_neighbours(current_feature,500,NN,features,mapping))
			
			same_track_beats=1
		else:	
			path = Path(get_track_from_name_and_number(last_beat_path,last_track_number+1))
			if(path.is_file()):
				appender.append(get_track_from_name_and_number(last_beat_path,last_track_number+1))
				last_track_number+=1
				same_track_beats+=1
				#print(t,get_track_from_name_and_number(last_beat_path,last_track_number+1))
				t=t+1
				
			
			else:	
				print("NAKO")
				last_beat_path=np.random.choice(mapping)
				last_track_name=get_track_name(last_beat_path)
				last_track_number=get_track_number(last_beat_path)
				same_track_beats=1
				#print(t,last_beat_path)
				t=t+1	
				

	output_file(appender)
	#feature=get_features_from_path(last_beat_path,mapping,features)	
generate(no_of_samples_in_remix,surprisal_factor,mapping,features,NN)
x_arr=np.arange(len(threshold_values))

plt.plot(x_arr,threshold_values,label="threshold_values")
plt.plot(x_arr,random_no,label="random_no")
plt.legend()
plt.show()
#print(first_beat_path)
#get_track_name(first_beat_path)
#get_track_number(first_beat_path)
#get_track_from_name_and_number(first_beat_path,2)
#print("FEATURES",get_features_from_path(first_beat_path,mapping,features))
