import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def shift_up(arr):
	return np.vstack([arr[1:], np.zeros((1, arr.shape[1]), int)])


def shift_down(arr):
	return np.vstack([np.zeros((1, arr.shape[1]), int), arr[:-1]])


def shift_left(arr):
	return np.hstack([arr[:, 1:], np.zeros((arr.shape[0], 1), int)])


def shift_right(arr):
	return np.hstack([np.zeros((arr.shape[0], 1), int), arr[:, :-1]])


def shift_up_left(arr):
	return shift_left(shift_up(arr))


def shift_up_right(arr):
	return shift_right(shift_up(arr))


def shift_down_left(arr):
	return shift_left(shift_down(arr))


def shift_down_right(arr):
	return shift_right(shift_down(arr))


def calc_neighbors(arr):
	return \
		shift_up(arr) + \
		shift_down(arr) + \
		shift_left(arr) + \
		shift_right(arr) + \
		shift_up_left(arr) + \
		shift_up_right(arr) + \
		shift_down_left(arr) + \
		shift_down_right(arr)


def build_glider(atx, aty, orientation):
	global life

	if orientation is 0:
		life[atx, aty] = 1
		life[atx + 2, aty] = 1
		life[atx + 1, aty + 1] = 1
		life[atx + 1, aty + 2] = 1
		life[atx + 2, aty + 1] = 1

	elif orientation is 1:
		life[atx + 2, aty] = 1
		life[atx, aty + 1] = 1
		life[atx + 1, aty + 1] = 1
		life[atx + 1, aty + 2] = 1
		life[atx + 2, aty + 2] = 1

	elif orientation is 2:
		life[atx + 1, aty] = 1
		life[atx + 2, aty] = 1
		life[atx, aty + 1] = 1
		life[atx + 1, aty + 1] = 1
		life[atx + 2, aty + 2] = 1

	elif orientation is 3:
		life[aty, atx] = 1
		life[aty + 2, atx] = 1
		life[aty + 1, atx + 1] = 1
		life[aty + 1, atx + 2] = 1
		life[aty + 2, atx + 1] = 1


import random as r

def build_init_life_stripes():

	global WIDTH, HEIGHT
	global life

	life = np.zeros((HEIGHT, WIDTH), dtype=int)

	for row in range(0,HEIGHT,2):
		for col in range(WIDTH):
			life[row,col] = 1

	for count in range(1000):
		life[r.randint(HEIGHT//4, 3*HEIGHT//4), r.randint(WIDTH//4, 3*WIDTH//4)] = 1

def build_init_life_stripes_2():

	global WIDTH, HEIGHT
	global life

	life = np.zeros((HEIGHT, WIDTH), dtype=int)

	for row in range(0,HEIGHT,2):
		for col in range(0,WIDTH):
			life[row,col] = 1

	for count in range(100):
		life[r.randint(HEIGHT//4, 3*HEIGHT//4), r.randint(WIDTH//4, 3*WIDTH//4)] = 1

def build_init_life_gliders():
	global WIDTH, HEIGHT

	global life
	life = np.zeros((HEIGHT, WIDTH), dtype=int)

	for row in range(HEIGHT//3, HEIGHT*2//3, 5):
		build_glider(row, row, 0)

	for row in range(2*WIDTH//3, WIDTH//2, -5):
		build_glider(row, HEIGHT - row, 1)

	# for count in range(100000):
	# 	life[r.randint(0, HEIGHT-1), r.randint(0, WIDTH-1)] = 1

# for row in range(200, 800):
# 	for col in range(200, 800):
# 		if row%2!=0 or col%2==0: life[row, col] = 1;

	print('go')


def build_init_life():
	global life
	life = np.zeros((HEIGHT, WIDTH), dtype=int)

	for count in range(4*HEIGHT//10):
		# life[r.randint(0, 99), r.randint(0, 99)] = 1
		life[HEIGHT//2 + count, WIDTH - 2 - count] = 1
		life[HEIGHT - 2 - count, WIDTH//2 + count] = 1
	# life[r.randint(0, 99), r.randint(0, 99)] = 1

	# for row in range(200, 800):
	# 	for col in range(200, 800):
	# 		if row%2!=0 or col%2==0: life[row, col] = 1;

	print('go')

def set_board_size(N,M):
	global HEIGHT
	global WIDTH

	HEIGHT = N
	WIDTH = M
	print ('setting board dimensions to: (%d,%d)' % (N,M))


def update_life():
	global life

	# print("generation %d:\n" % frame_number, life)

	nbors = calc_neighbors(life)

	# print("nbors %d:\n" % frame_number, nbors)

	life = ((life & ((nbors == 2) | (nbors == 3))) + ((~life) & (nbors == 3)))
	# input("ENTER to continue:")

def init():
	im.set_data(life)
	return im

def animate(i):
	global im
	global generation

	update_life()
	generation += 1
	if generation % 100 == 0: print(generation)

	im.set_data(life)
	return im

global life
global im
global generation

generation = 0
set_board_size(1000,1000)
#build_init_life()
# build_init_life_gliders()

# build_init_life_stripes()
build_init_life_stripes_2()

# plt.style.use('classic')
fig = plt.figure()
im = plt.imshow(life) #, cmap='gist_gray_r')


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=2000,
							   interval=10)

plt.show()
