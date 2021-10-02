#!/usr/bin/python
# -*- coding: utf-8 -*-

# Programa simples com camera webcam e opencv

import cv2
import math
from datetime import datetime
import numpy as np
from numpy.core.fromnumeric import size
from helper_functions import calc_hsv_range, find_greater, draw_cross, draw_text

def image_da_webcam(img):
	"""
	->>> !!!! FECHE A JANELA COM A TECLA ESC !!!! <<<<-
		deve receber a imagem da camera e retornar uma imagems filtrada.
	"""  

	#region Carregamento da imagem

	# Um pouco de blur para reduzir ruídos
	imgb = cv2.blur(img, (11, 11))

	img_rgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2RGB)
	img_hsv = cv2.cvtColor(imgb, cv2.COLOR_BGR2HSV)
	img_size = img_rgb.shape

	#endregion

	#region Segmentação dos círculos

	# Valores HSV aproximados para os círculos azuis e vermelhos
	blue_hsv = (189, 42, 42)
	red_hsv = (2, 60, 38)

	blue_hsv = (187, 85, 70)
	red_hsv = (3, 70, 70)

	# Threshold para segmentação da imagem
	threshold = (8, 50, 50)

	# Ranges mínimos e máximos da imagem, recebe valores HSV entre
	# [(0, 0, 0), (360, 100, 100)] e retorna valores HSV para opencv entre
	# [(0, 0, 0), (180, 255, 255)]
	# Retorna dois ranges, um para a parte baixa do range e outra para a parte
	# alta do range
	range_blue = calc_hsv_range(blue_hsv, threshold)
	range_red = calc_hsv_range(red_hsv, threshold)

	# Segmenta os dois ranges dos círculos azuis
	mask_blue_1 = cv2.inRange(img_hsv, range_blue[0], range_blue[1])
	mask_blue_2 = cv2.inRange(img_hsv, range_blue[2], range_blue[3])

	# Segmenta os dois ranges dos círculos vermelhos
	mask_red_1 = cv2.inRange(img_hsv, range_red[0], range_red[1])
	mask_red_2 = cv2.inRange(img_hsv, range_red[2], range_red[3])

	# Junta as imagens segmentadas
	mask_blue = cv2.bitwise_or(mask_blue_1, mask_blue_2)
	mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

	#endregion

	#region Acha os contornos nas segmentações
	
	cont_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cont_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	# Converte para imagens RGB
	blue_rgb = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2RGB)
	red_rgb = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2RGB)

	# Cria uma cópia da imagem original
	cont_img = img.copy()

	# Desenha os contornos dos dois círculos
	# cv2.drawContours(cont_img, cont_blue, -1, [0, 0, 255], 5)
	# cv2.drawContours(cont_img, cont_red, -1, [0, 0, 255], 5)

	#endregion

	#region Identificação dos maiores círculos

	found_greater_blue, greater_blue, greater_blue_area = find_greater(cont_blue)
	found_greater_red, greater_red, greater_red_area = find_greater(cont_red)

	if not found_greater_blue or not found_greater_red: return cont_img

	# Desenha os contornos dos dois círculos maiores
	cv2.drawContours(cont_img, [greater_blue], -1, [0, 125, 0], 5)
	cv2.drawContours(cont_img, [greater_red], -1, [0, 125, 0], 5)

	#endregion

	#region Identificação do centro de massa dos dois círculos

	moments_blue = cv2.moments(greater_blue)
	moments_red = cv2.moments(greater_red)

	# Centro de massa do círculo azul
	blue_center_x = int(moments_blue['m10']/moments_blue['m00'])
	blue_center_y = int(moments_blue['m01']/moments_blue['m00'])
	draw_cross(cont_img, (blue_center_x, blue_center_y), (10, 10), (0, 0, 0), 4)

	# Centro de massa do círculo vermelho
	red_center_x = int(moments_red['m10']/moments_red['m00'])
	red_center_y = int(moments_red['m01']/moments_red['m00'])
	draw_cross(cont_img, (red_center_x, red_center_y), (10, 10), (0, 0, 0), 4)

	#endregion

	#region Identificação da área dos dois círculos

	# Desenha a área do círculo azul
	draw_text(
		cont_img, (blue_center_x + 16, blue_center_y - 16),
		cv2.FONT_HERSHEY_SIMPLEX, int(greater_blue_area), (0, 0, 0)
	)

	# Desenha a área do círculo vermelho
	draw_text(
		cont_img, (red_center_x + 16, red_center_y - 16),
		cv2.FONT_HERSHEY_SIMPLEX, int(greater_red_area), (0, 0, 0)
	)

	#endregion

	#region Traça uma linha entre os dois círculos maiores

	# Simplesmente desenha uma linha
	cv2.line(
		cont_img, (blue_center_x, blue_center_y),(red_center_x, red_center_y),
		(125, 0, 125), 4
	)

	#endregion

	#region Ângulo entre os círculos

	# Calcula diferença entre as posições dos círculos
	diff_y = (red_center_x - blue_center_x)
	diff_y = (red_center_y - blue_center_y)

	# Centro entre os círculos
	center_x = int((red_center_x + blue_center_x) * .5)
	center_y = int((red_center_y + blue_center_y) * .5)

	# Calcula o ângulo em radianos usando arcotangente
	ang_rad_z = math.atan2(diff_y, diff_y)

	# Converte de radianos para graus 
	ang_deg_z = ang_rad_z * (180.0 / math.pi)

	# Ângulo no eixo y
	# diff_y = (greater_red_area - greater_blue_area)

	# ang_y = math.atan2(diff_y / max(greater_blue_area, greater_red_area), 1)

	# Desenha a reta de origem
	cv2.line(
		cont_img, (0, center_y),(img_size[1], center_y),
		(0, 0, 0), 4
	)

	# Desenha um arco do ângulo
	cv2.ellipse(
		cont_img, (center_x, center_y), (150, 150), 0, 0, ang_deg_z,
		(150, 64, 42), 4
	)

	# Desenha o texto do ângulo
	draw_text(
		cont_img, (center_x + 150, center_y - 16 if ang_deg_z > 0.0 else center_y + 32),
		cv2.FONT_HERSHEY_SIMPLEX, str(int(ang_deg_z)) + "d", (0, 0, 0)
	)

	#endregion

	return cont_img

cv2.namedWindow("original", cv2.WINDOW_NORMAL)
cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
# define a entrada de video para webcam
vc = cv2.VideoCapture(0)
#vc = cv2.VideoCapture("C:\\Users\\Administrator\\Documents\\Codes\\Notebooks\\NAC 5\\Demo.mp4")

#configura o tamanho da janela 
cv2.resizeWindow("original", 640, 480)
cv2.resizeWindow("preview", 640, 480)

if vc.isOpened(): # try to get the first frame
	rval, frame = vc.read()
else:
	rval = False

while rval:
	img = image_da_webcam(frame) # passa o frame para a função imagem_da_webcam e recebe em img imagem tratada

	#imS = cv2.resize(im, (960, 540))

	cv2.imshow("original", frame)
	cv2.imshow("preview", img)
	rval, frame = vc.read()
	key = cv2.waitKey(20)
	if key == 27: # exit on ESC
		break

cv2.destroyWindow("original")
cv2.destroyWindow("preview")
vc.release()
