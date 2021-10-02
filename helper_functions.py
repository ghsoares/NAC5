import numpy as np	
import cv2

# Essa função calcula o range hsv mínimo e máximo
# a partir de uma cor hsv da parte baixa e alta do filtro.
# Retorna dois ranges para poder cobrir valores
# extremos tipo 0 ou 349 para hue
def calc_hsv_range(hsv, threshold: int):
	h = hsv[0]
	s = hsv[1]
	v = hsv[2]

	threshold_h = threshold[0]
	threshold_s = threshold[1]
	threshold_v = threshold[2]

	# converte H de [0-360] para [0-180]
	h_min = int((h - threshold_h) * .5)
	# converte S de [0-100] para [0-255]
	s_min = int((s - threshold_s) * 2.55)
	# converte V de [0-100] para [0-255]
	v_min = int((v - threshold_v) * 2.55)

	h_max = int((h + threshold_h) * .5)
	s_max = int((s + threshold_s) * 2.55)
	v_max = int((v + threshold_v) * 2.55)

	# limita o range [0-255] para S e V
	s_min = max(0, min(s_min, 255))
	s_max = max(0, min(s_max, 255))
	v_min = max(0, min(v_min, 255))
	v_max = max(0, min(v_max, 255))

	# Listas
	range_min_1, range_min_2 = [h_min, s_min, v_min], [h_min, s_min, v_min]
	range_max_1, range_max_2 = [h_max, s_max, v_max], [h_max, s_max, v_max]

	# Cobre o H para ranges fora de [0-180]
	if h_min < 0:
		range_min_1[0] = 179 + h_min
		range_max_1[0] = 179
		range_min_2[0] = 0
	if h_max > 179:
		range_max_1[0] = 179
		range_min_2[0] = 0
		range_max_2[0] = h_max - 179

	# Retorna quatro arrays, dois da parte baixa do range
	# e dois da parte alta do range
	return (
		np.array([range_min_1[0], range_min_1[1], range_min_1[2]]),
		np.array([range_max_1[0], range_max_1[1], range_max_1[2]]),
		np.array([range_min_2[0], range_min_2[1], range_min_2[2]]),
		np.array([range_max_2[0], range_max_2[1], range_max_2[2]])
	)

# Essa função acha o maior contorno entre os contornos fornecidos
def find_greater(contours):
	greater = None
	greater_area = 0
	found = False
	for c in contours:
		area = cv2.contourArea(c)
		if area > greater_area:
			greater_area = area
			greater = c
			found = True
	return found, greater, greater_area

# Essa função desenha uma cruz na imagem
def draw_cross(img, center, size, color, width = 5):
	cv2.line(
		img,
		(center[0] - size[0], center[1]),
		(center[0] + size[0], center[1]),
		color, width
	)
	cv2.line(
		img,
		(center[0], center[1] - size[1]),
		(center[0], center[1] + size[1]),
		color, width
	)

# Essa função desenha um texto na imagem
def draw_text(img, center, font, text, color, font_scale = 1, font_thickness = 2):
	cv2.putText(
		img, str(text), center,
		font, font_scale, color, font_thickness,
		cv2.LINE_AA
	)


