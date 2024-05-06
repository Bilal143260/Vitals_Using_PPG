import gradio as gr
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

#TODO: change function name
#TODO: add separate function to calculate bpm and also create master function to call all functions

def find_bpm(video):
    video = imageio.get_reader(video, 'ffmpeg')

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
    # print('Estimated Heartate: {} bpm'.format(heartrate))

    #Get breathing rate
    heart_rate = find_breathing_rate(colors['red'])

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(16,9))
    plt.plot(x_filt, colors['red_filt'], color='#fc4f30')
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Pixel Color')
    plt.title('Filtered Red Channel Pixel Data')
    fig2 = plt.gcf()
    # fig2.savefig('filtered.png', dpi=200)

    return heartrate, heart_rate, fig2

def find_breathing_rate(red_channel, fps=60):
    # Assume red_channel is already a 1D numpy array of normalized red values
    
    # Apply a low-pass filter to focus on breathing rate frequencies (0.1 to 0.4 Hz)
    filtered_signal = apply_low_pass_filter(red_channel, fps)
    
    # Calculate FFT of the filtered signal
    fft_result = np.fft.rfft(filtered_signal)
    freqs = np.fft.rfftfreq(len(filtered_signal), d=1/fps)
    
    # Identify the frequency with the maximum amplitude in the breathing rate range
    breathing_rate_freq = extract_breathing_rate_freq(fft_result, freqs)
    
    # Calculate breathing rate in breaths per minute
    breathing_rate_bpm = breathing_rate_freq * 60
    
    return breathing_rate_bpm


def apply_low_pass_filter(signal, fps, cutoff_hz=0.4):

    # Calculate the Nyquist frequency
    nyq = 0.5 * fps
    normal_cutoff = cutoff_hz / nyq

    # Butterworth low-pass filter
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    filtered_output = filtfilt(b, a, signal)
    return filtered_output

def extract_breathing_rate_freq(fft_result, freqs):
    # Define breathing rate frequency range in Hz
    lower_bound = 0.1
    upper_bound = 0.4

    # Find indices where frequency is within the breathing rate range
    indices = (freqs >= lower_bound) & (freqs <= upper_bound)

    # Extract the corresponding FFT values and frequencies
    relevant_freqs = freqs[indices]
    relevant_fft_values = np.abs(fft_result[indices])

    # Find the frequency with the maximum FFT value in the range
    max_index = np.argmax(relevant_fft_values)
    dominant_freq = relevant_freqs[max_index]
    return dominant_freq

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_video = gr.PlayableVideo(format="mp4", interactive=True, 
                                           autoplay=True, width="600px", height="400px")
        with gr.Column():
            # graphs = gr.Gallery(preview=True, object_fit="scale-down", interactive=False)
            plot_output = gr.Plot()
            bpm = gr.Textbox(label="BMP")
            heart_rate = gr.Textbox(label="BREATHING RATE")
    
    with gr.Row():
        with gr.Column():
            generate = gr.Button()
    
    generate.click(fn=find_bpm,inputs=[input_video],outputs=[bpm,heart_rate,plot_output])
    # generate_heart_rate.click(fn=find_breathing_rate,inputs=[input_video],outputs=[heart_rate])

if __name__ == "__main__":
    demo.launch()