# ------------------------------------------------------------------------------
# Copyright (c) 
# Createdd by Yan
# Data:2021
# ------------------------------------------------------------------------------

import cv2
import numpy as np
import random
import os
import sys


def play_multi_videos(width, height, i, vid_1_path, vid_2_path, vid_3_path, vid_4_path, vid_5_path, vid_6_path):


	print("-"*10)
	print("in function")

	print(vid_1_path)
	print(vid_2_path)
	print(vid_3_path)
	print(vid_4_path)
	print(vid_5_path)
	print(vid_6_path)

	print("-"*10)


	vid_1 = cv2.VideoCapture(vid_1_path)
	vid_2 = cv2.VideoCapture(vid_2_path)
	vid_3 = cv2.VideoCapture(vid_3_path)
	vid_4 = cv2.VideoCapture(vid_4_path)
	vid_5 = cv2.VideoCapture(vid_5_path)
	vid_6 = cv2.VideoCapture(vid_6_path)



	# frame per second
	fps = vid_1.get(cv2.CAP_PROP_FPS)

	# window name displayed 
	window_name = 'Blind_Evaluation'
  
	# caption font 
	font = cv2.FONT_HERSHEY_SIMPLEX 
	  
	# caption location 
	org = (450, 30) 
	  
	# fontScale 
	fontScale = 1
	   
	# green color in BGR 
	color = (0, 255, 0) 
	  
	# Line thickness of 2 px 
	thickness = 2

	# codec for mp4
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out_video = cv2.VideoWriter('./output/video{:03d}.mp4'.format(i+1), fourcc, fps, (width*3, height*2))

	while True:

		(grabbed_1, frame_1) = vid_1.read()
		(grabbed_2, frame_2) = vid_2.read()
		(grabbed_3, frame_3) = vid_3.read()
		(grabbed_4, frame_4) = vid_4.read()
		(grabbed_5, frame_5) = vid_5.read()
		(grabbed_6, frame_6) = vid_6.read()

		if grabbed_1 == True & grabbed_2 == True & grabbed_3 == True & grabbed_4 == True & grabbed_5 == True & grabbed_6 == True:

			frame_resized_1 = cv2.resize(frame_1, (width, height), interpolation=cv2.INTER_CUBIC)
			frame_resized_2 = cv2.resize(frame_2, (width, height), interpolation=cv2.INTER_CUBIC)
			frame_resized_3 = cv2.resize(frame_3, (width, height), interpolation=cv2.INTER_CUBIC)
			frame_resized_4 = cv2.resize(frame_4, (width, height), interpolation=cv2.INTER_CUBIC)
			frame_resized_5 = cv2.resize(frame_5, (width, height), interpolation=cv2.INTER_CUBIC)
			frame_resized_6 = cv2.resize(frame_6, (width, height), interpolation=cv2.INTER_CUBIC)


			# Using cv2.putText() method 
			frame_resized_1 = cv2.putText(frame_resized_1, 'A', org, font, fontScale, color, thickness, cv2.LINE_AA)
			frame_resized_2 = cv2.putText(frame_resized_2, 'B', org, font, fontScale, color, thickness, cv2.LINE_AA)
			frame_resized_3 = cv2.putText(frame_resized_3, 'C', org, font, fontScale, color, thickness, cv2.LINE_AA)
			frame_resized_4 = cv2.putText(frame_resized_4, 'D', org, font, fontScale, color, thickness, cv2.LINE_AA)
			frame_resized_5 = cv2.putText(frame_resized_5, 'E', org, font, fontScale, color, thickness, cv2.LINE_AA)
			frame_resized_6 = cv2.putText(frame_resized_6, 'F', org, font, fontScale, color, thickness, cv2.LINE_AA)

			# put videos into 3x2 grid
			img_stack_1 = np.hstack((frame_resized_1, frame_resized_2))
			img_stack_1 = np.hstack((img_stack_1, frame_resized_3))


			img_stack_2 = np.hstack((frame_resized_4, frame_resized_5))
			img_stack_2 = np.hstack((img_stack_2, frame_resized_6))

			img_stack_3 = np.vstack((img_stack_1,img_stack_2))

			cv2.imshow(window_name, img_stack_3)
			# print('*'*20)
			# print(img_stack_3[0])

			# write video
			# out_video = cv2.VideoWriter('./output/video{:03d}.mp4'.format(i+1), fourcc, fps, (width*3, height*2))
			out_video.write(img_stack_3)

			if cv2.waitKey(10)==ord('q'):
				break

		else:
			print("No video(s) is to be played, please make sure it/they are here")
			vid_1.release()
			vid_2.release()
			vid_3.release()
			vid_4.release()
			vid_5.release()
			vid_6.release()
			cv2.destroyAllWindows()
			break

if __name__ == '__main__':

	video_path = './blindeval/skegho_hrnetbu_maskrcnn/'
	videos = os.listdir(video_path) # all the video names

	# make directory for output
	if not os.path.exists('output'):
		os.mkdir('output')

	# write video list into txt file
	f = open('./output/video_info.txt', 'w+')
	f.write('Original video name || New output video name || Used method list [A, B, C, D, E, F] \n')

	num_video = len(videos) # number of videos, 30
	num_video = 30 #1
	for i in range(num_video):

		print("="*10)
		print("="*3+" "*3 +str(i)+ " "*3 + "="*3)
		print("="*10)


		# specify video path
		vid_1_path = 'blindeval/skegho_hrnetbu_maskrcnn/' + videos[i]
		vid_2_path = 'blindeval/skegho_openpose_maskrcnn/' + videos[i]
		vid_3_path = 'blindeval/skegho_udp_maskrcnn/' + videos[i]
		vid_4_path = 'blindeval/skegho_hrnetbu_solov2/' + videos[i]
		vid_5_path = 'blindeval/skegho_openpose_solov2/' + videos[i]
		vid_6_path = 'blindeval/skegho_udp_solov2/' + videos[i]


		# random video list
		video_method_list = ['vid_1_path', 'vid_2_path', 'vid_3_path', 'vid_4_path', 'vid_5_path', 'vid_6_path']
		# print(video_method_list)

		print("="*10)
		print("Before shuffling")

		for item in video_method_list:
			print(eval(item))

		random.shuffle(video_method_list)
		video_method_list_shuffle = video_method_list

		# set the resolution for each video
		width, height = 480, 270

		vid_1_path_shuffle = eval(video_method_list_shuffle[0])
		vid_2_path_shuffle = eval(video_method_list_shuffle[1])
		vid_3_path_shuffle = eval(video_method_list_shuffle[2])
		vid_4_path_shuffle = eval(video_method_list_shuffle[3])
		vid_5_path_shuffle = eval(video_method_list_shuffle[4])
		vid_6_path_shuffle = eval(video_method_list_shuffle[5])

		print("="*10)

		print("After shuffling")

		print(vid_1_path_shuffle)
		print(vid_2_path_shuffle)
		print(vid_3_path_shuffle)
		print(vid_4_path_shuffle)
		print(vid_5_path_shuffle)
		print(vid_6_path_shuffle)

		print("="*10)

		# sys.exit(1)


		method_list = ['hrnetbu_maskrcnn', 'openpose_maskrcnn', 'udp_maskrcnn', 'hrnetbu_solov2', 'openpose_solov2', 'udp_solov2']
		method_1 = vid_1_path_shuffle.split('/')[1].split('_')[1] + "_" + vid_1_path_shuffle.split('/')[1].split('_')[2]
		method_2 = vid_2_path_shuffle.split('/')[1].split('_')[1] + "_" + vid_2_path_shuffle.split('/')[1].split('_')[2]
		method_3 = vid_3_path_shuffle.split('/')[1].split('_')[1] + "_" + vid_3_path_shuffle.split('/')[1].split('_')[2]
		method_4 = vid_4_path_shuffle.split('/')[1].split('_')[1] + "_" + vid_4_path_shuffle.split('/')[1].split('_')[2]
		method_5 = vid_5_path_shuffle.split('/')[1].split('_')[1] + "_" + vid_5_path_shuffle.split('/')[1].split('_')[2]
		method_6 = vid_6_path_shuffle.split('/')[1].split('_')[1] + "_" + vid_6_path_shuffle.split('/')[1].split('_')[2]


		print("#"*10)
		print(method_1)

		method_list = [method_1, method_2, method_3, method_4, method_5, method_6]

		print("#"*10)
		print(method_list)

		# continue to write video list content
		f.write(videos[i] + '|| video{:03d}.mp4'.format(i+1) + '|| [' + ", ".join(method_list) + ']\n')		
		# vid_method = vid_1_path.split('/')[1]
		f.flush()


		play_multi_videos(width, height, i, vid_1_path_shuffle,vid_2_path_shuffle,vid_3_path_shuffle,vid_4_path_shuffle,vid_5_path_shuffle,vid_6_path_shuffle)

	f.close()
