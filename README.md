# Video Style Transfer
Executing style transfer to a content video and a style image.

# Progress - 1st Week:
Project outline: style transfer in real time


## Before you run: 

1) Highly recommend running this code in cuda server, which can use GPU rather than CPU  - because we're doing image.
2) Install pip3 with opencv - because we're getting videostream
3) Compile:

  $ python3 videoTransfer.py
  

## Idea: 
Model Compression => to reduce time for each frame processing

The key idea was receiving the video and read each of the frame to manipulate. After getting the video, save into the frame by fame array and append everything to work as how the static style transfer has been done. There are more areas to improve - such as optimizing the video processing. What I meant by reducing the time can be done by dropping some of the unnecessary frames by reading it. The meaning of unncessary frames mean that some of the frames in the video do not have a much of movement, which do not give much new information compared to the previous frame. Therefore, we're aware of the fact that there are ways to reduce 


## Time lines:

1) 11/16-17: Reading articles and make sure we have fullly understanding about those models
2) 11/18-20: Making sure done with real time video processing and small test set example,
11/20: Visit office hours to make sure we're on the track
3) 11/21-25: Make sure everytihng can run and make small two test cases individually. 


## References:
https://towardsdatascience.com/real-time-video-neural-style-transfer-9f6f84590832

https://medium.com/coinmonks/real-time-video-style-transfer-fast-accurate-and-temporally-consistent-863a175e06dc

https://blog.paperspace.com/creating-your-own-style-transfer-mirror/
