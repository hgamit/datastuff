import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

global life


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


def build_init_life():
    global life
    life = np.zeros((30, 30), dtype=int)
    for row in range(23, 28):
        for col in range(23, 28):
            life[row, col] = 1;


def gen(n):
    global life
    for count in range(n):
        print("generation %d:\n" % count, life)

        nbors = calc_neighbors(life)

        print("nbors %d:\n" % count, nbors)

        life = ((life & ((nbors == 2) | (nbors == 3))) + ((~life) & (nbors == 3)))


# def run_life(iter):
#
# 	import time
# 	# for count in range(iter):
# 	# 	# plt.imshow(brd, origin='upper', extent=[0, brd.shape[0], 0, brd.shape[1]],
# 		# 		   cmap='viridis')
# 		# plt.colorbar();
# 		#         nbors = calc_neighbors(brd)
# 		#         new_brd = ((brd & ((nbors==2)|(nbors==3)))+((~brd) & (nbors==3)))
# 		#         brd = new_brd
# 	gen(iter)
# 		# time.sleep(0.1)

# def run_anim():
# 	# from matplotlib.org site on Animation
#
# 	fig = plt.figure(figsize=(7, 7))
# 	ax = fig.add_axes([0, 0, 1, 1], frameon=False)
# 	ax.set_xlim(0, 1), ax.set_xticks([])
# 	ax.set_ylim(0, 1), ax.set_yticks([])
#
# 	# Create rain data
# 	global rain_drops
# 	global n_drops
#
# 	n_drops = 50
# 	rain_drops = np.zeros(n_drops, dtype=[('position', float, 2),
# 										  ('size', float, 1),
# 										  ('growth', float, 1),
# 										  ('color', float, 4)])
#
# 	# Initialize the raindrops in random positions and with
# 	# random growth rates.
# 	rain_drops['position'] = np.random.uniform(0, 1, (n_drops, 2))
# 	rain_drops['growth'] = np.random.uniform(50, 200, n_drops)
#
# 	# Construct the scatter which we will update during animation
# 	# as the raindrops develop.
# 	global scat
# 	scat = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1],
# 					  s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
# 					  facecolors='none')
# 	animation = FuncAnimation(fig, update, interval=10)
# 	plt.show()

# def update(frame_number):
#     # Get an index which we can use to re-spawn the oldest raindrop.
#     current_index = frame_number % n_drops
#
#     # Make all colors more transparent as time progresses.
#     rain_drops['color'][:, 3] -= 1.0/len(rain_drops)
#     rain_drops['color'][:, 3] = np.clip(rain_drops['color'][:, 3], 0, 1)
#
#     # Make all circles bigger.
#     rain_drops['size'] += rain_drops['growth']
#
#     # Pick a new position for oldest rain drop, resetting its size,
#     # color and growth factor.
#     rain_drops['position'][current_index] = np.random.uniform(0, 1, 2)
#     rain_drops['size'][current_index] = 5
#     rain_drops['color'][current_index] = (0, 0, 0, 1)
#     rain_drops['growth'][current_index] = np.random.uniform(50, 200)
#
#     # Update the scatter collection, with the new colors, sizes and positions.
#     scat.set_edgecolors(rain_drops['color'])
#     scat.set_sizes(rain_drops['size'])
#     scat.set_offsets(rain_drops['position'])

def update_life():
    # # Get an index which we can use to re-spawn the oldest raindrop.
    # current_index = frame_number % n_drops
    #
    # # Make all colors more transparent as time progresses.
    # rain_drops['color'][:, 3] -= 1.0/len(rain_drops)
    # rain_drops['color'][:, 3] = np.clip(rain_drops['color'][:, 3], 0, 1)
    #
    # # Make all circles bigger.
    # rain_drops['size'] += rain_drops['growth']
    #
    # # Pick a new position for oldest rain drop, resetting its size,
    # # color and growth factor.
    # rain_drops['position'][current_index] = np.random.uniform(0, 1, 2)
    # rain_drops['size'][current_index] = 5
    # rain_drops['color'][current_index] = (0, 0, 0, 1)
    # rain_drops['growth'][current_index] = np.random.uniform(50, 200)
    #
    # # Update the scatter collection, with the new colors, sizes and positions.
    # scat.set_edgecolors(rain_drops['color'])
    # scat.set_sizes(rain_drops['size'])
    # scat.set_offsets(rain_drops['position'])

    global life

    # print("generation %d:\n" % frame_number, life)

    nbors = calc_neighbors(life)

    # print("nbors %d:\n" % frame_number, nbors)

    life = ((life & ((nbors == 2) | (nbors == 3))) + ((~life) & (nbors == 3)))


# line.set_data(life)
# return line,


# def run_life_anim():
# 	# from matplotlib.org site on Animation
# 	fig = plt.figure(figsize=(30, 30),facecolor='black')
#
# 	# ax = fig.add_axes([0, 0, 1, 1], frameon=False)
# 	plt.imshow(life, origin='upper', extent=[0, life.shape[0], 0, life.shape[1]],
# 			   cmap='viridis')
#
# 	# ax.set_xlim(0, 1), ax.set_xticks([])
# 	# ax.set_ylim(0, 1), ax.set_yticks([])
#
# 	# ax.set_xlim(0, 1), ax.set_xticks([])
# 	# ax.set_ylim(0, 1), ax.set_yticks([])
#
# 	# ax.set_ylim(0, 30)
# 	# ax.set_xlim(0, 30)
# 	#
# 	# # No ticks
# 	# ax.set_xticks([])
# 	# ax.set_yticks([])
#
#
# 	global life
#
# 	# ax.plot(life, color="bs")
#
# 	# Set y limit (or first line is cropped because of thickness)
#
# 	# Construct the scatter which we will update during animation
# 	# as the raindrops develop.
# 	global plot
# 	# plt.plot(life,'ks')
# 	# plot = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1],
# 	# 				  s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
# 	# 				  facecolors='none')
#
# 	animation = FuncAnimation(fig, update_life, interval=10)
# 	plt.show()

# def init():
# 	ax.set_xlim(0, 30)
# 	ax.set_ylim(0, 3 )
# 	return ln,
#
# def update(frame):
# 	xdata.append(frame)
# 	ydata.append(np.sin(frame))
# 	ln.set_data(xdata, ydata)
# 	return ln,

# def run_life_anim_2():
# 	global life
# 	global ln
# 	global ax
#
#
# 	fig, ax = plt.subplots()
# 	# xdata, ydata = [], []
# 	# ln, = plt.plot(life, 'sk', animated=True)
#
# 	plt.imshow(life, origin='upper', extent=[0, life.shape[0], 0, life.shape[1]],
# 			   cmap='viridis')
#
# 	ani = FuncAnimation(fig, update_life, frames=life) #,
# 						# init_func=init, blit=True)
# 	plt.show()

def build_init_life():
    global life
    life = np.zeros((1000, 1000), dtype=int)
    for row in range(10, 20):
        for col in range(10, 28):
            life[row, col] = 1;


# def main():
#
# 	global life
#
# 	build_init_life()
#
# 	# print(life, '\n')
# 	# print(calc_neighbors(life))
#
# 	# run_life(20)
#
# 	# run_life_anim()
#
# main()

# run_anim()

# build_init_life()
#
# print (life)
#
# gen(7)

global life
# global ax
# global line

# plt.imshow(life, origin='upper', extent=[0, life.shape[0], 0, life.shape[1]],
# 		   cmap='viridis')
# plt.show()
#
# fig = plt.figure()
# # ax = plt.axes(xlim=(0, 30), ylim=(0, 30))
# # line,
# ax = plt.imshow(life, origin='upper', extent=[0, life.shape[0], 0, life.shape[1]],
# 		   cmap='viridis')
#
#
# # initialization function: plot the background of each frame
# def init():
#     line.set_data(life)
#     return line,
#
# # animation function.  This is called sequentially
# def animate(i):
#     line.set_data(life)
#     return line, # should be stripped
#
# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = FuncAnimation(fig, update_life, init_func=init, interval=20, blit=True)
# plt.show()

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

nx = 30
ny = 30
global im
build_init_life()

fig = plt.figure()
im = plt.imshow(life, cmap='gist_gray_r')


def init():
    im.set_data(life)


def animate(i):
    global im
    update_life()
    im.set_data(life)
    return im


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1000,
                               interval=10)

plt.show()
