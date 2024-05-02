import numpy as np
import imageio

video = imageio.get_reader('test_video.mp4', 'ffmpeg')

#data extraction for each channel
colors = {'red': [], 'green': [], 'blue': []}
for frame in video:
    # Average all pixels
    lumped_pixel = np.mean(frame, axis=(0,1))
    colors['red'].append(lumped_pixel[0])
    colors['green'].append(lumped_pixel[1])
    colors['blue'].append(lumped_pixel[2])

#normalization
for key in colors:
    colors[key] = np.divide(colors[key], 255)

fps = 60 # frames-per-second from video
x = np.arange(len(colors['red'])) / fps

#filtering
colors['red_filt'] = list()
colors['red_filt'] = np.append(colors['red_filt'], colors['red'][0])
tau = 0.25 # HPF time constant in seconds
fsample = fps # Sample rate
alpha = tau / (tau + 2/fsample)
for index, frame in enumerate(colors['red']):
    if index > 0:
        y_prev = colors['red_filt'][index - 1]
        x_curr = colors['red'][index]
        x_prev = colors['red'][index - 1]
        colors['red_filt'] = np.append(colors['red_filt'], alpha * (y_prev + x_curr - x_prev))

# Want to truncate data since beginning of series will be wonky
x_filt = x[50:-1]
colors['red_filt'] = colors['red_filt'][50:-1]

#fft
red_fft = np.absolute(np.fft.fft(colors['red_filt']))
N = len(colors['red_filt'])
freqs = np.arange(0,fsample/2,fsample/N)

# Truncate to fs/2
red_fft = red_fft[0:len(freqs)]

# Get heartrate from FFT
max_val = 0
max_index = 0
for index, fft_val in enumerate(red_fft):
    if fft_val > max_val:
        max_val = fft_val
        max_index = index

heartrate = round(freqs[max_index] * 60,1)
print('Estimated Heartate: {} bpm'.format(heartrate))