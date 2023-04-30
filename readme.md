First create beat segments using divide_beat_segments.py it takes two arguments
- **path of directory containing songs to remi**
-  **output directory containing beat segments**

Ex Usage: python3 divide_beat_segments.py mysongs2/ /my_beats_2

In extraction directory, run extractor.py it takes four arguments
- **directory containing beat segments of songs to remix**
- **path of mapping pkl file**
- **path of dataset containing features pkl file**
- **path of NearestNeighbour model pkl file**

Ex Usage: python3 extractor.py ../my_beats_2/ mapping.pkl dataset.pkl nn.pkl


Adjust the names and path of pkl files in create_remix.py from lines 107-112, this program takes two arguments
- **number of beat segments in remix**
- **Surprisal Factor** (recommended values between around 0.02)

Ex Usage: python3 create_remix.py 80 0.019



