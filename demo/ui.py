import gradio as gr
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def extract_color_data(video):
    colors = {'red': [], 'green': [], 'blue': []}
    for frame in video:
        lumped_pixel = np.mean(frame, axis=(0, 1))
        colors['red'].append(lumped_pixel[0])
        colors['green'].append(lumped_pixel[1])
        colors['blue'].append(lumped_pixel[2])
    for key in colors:
        colors[key] = np.divide(colors[key], 255)
    return colors

def filter_signal(signal, fps):
    tau = 0.25
    alpha = tau / (tau + 2 / fps)
    filtered_signal = [signal[0]]
    for index in range(1, len(signal)):
        y_prev = filtered_signal[index - 1]
        x_curr = signal[index]
        x_prev = signal[index - 1]
        filtered_signal.append(alpha * (y_prev + x_curr - x_prev))
    return filtered_signal[50:], np.arange(len(filtered_signal))[50:] / fps

def calculate_fft(signal, fps):
    fft_result = np.abs(np.fft.fft(signal))
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1 / fps)[:N // 2]
    fft_result = fft_result[:N // 2]
    return freqs, fft_result

def find_bpm(signal, fps):
    freqs, fft_result = calculate_fft(signal, fps)
    bpm_freq = freqs[np.argmax(fft_result)]
    return round(bpm_freq * 60, 1)

def apply_low_pass_filter(signal, fps, cutoff_hz=0.4):
    nyq = 0.5 * fps
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def extract_breathing_rate_freq(fft_result, freqs):
    lower_bound = 0.1
    upper_bound = 0.4
    indices = (freqs >= lower_bound) & (freqs <= upper_bound)
    relevant_freqs = freqs[indices]
    relevant_fft_values = np.abs(fft_result[indices])
    dominant_freq = relevant_freqs[np.argmax(relevant_fft_values)]
    return dominant_freq

def find_breathing_rate(signal, fps):
    filtered_signal = apply_low_pass_filter(signal, fps)
    freqs, fft_result = calculate_fft(filtered_signal, fps)
    breathing_rate_freq = extract_breathing_rate_freq(fft_result, freqs)
    return round(breathing_rate_freq * 60, 1)

def calculate_spo2(red_signal, green_signal):
    # Simplified example assuming direct correlation for demonstration purposes
    spo2 = 100 - 5 * (np.std(green_signal) / np.std(red_signal))
    return round(spo2, 1)

def analyze_video(video):
    video = imageio.get_reader(video, 'ffmpeg')
    colors = extract_color_data(video)
    
    fps = 60
    filtered_red, x_filt = filter_signal(colors['red'], fps)
    
    bpm = find_bpm(filtered_red, fps)
    breathing_rate = find_breathing_rate(colors['red'], fps)
    spo2 = calculate_spo2(colors['red'], colors['green'])

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(16, 9))
    plt.plot(x_filt, filtered_red, color='#fc4f30')
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Pixel Color')
    plt.title('Filtered Red Channel Pixel Data')
    fig2 = plt.gcf()

    return int(bpm), int(breathing_rate), int(spo2), fig2

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_video = gr.PlayableVideo(format="mp4", interactive=True, autoplay=True, width="600px", height="400px")
        with gr.Column():
            plot_output = gr.Plot()
            bpm = gr.Textbox(label="BPM")
            breathing_rate = gr.Textbox(label="Breathing Rate")
            spo2 = gr.Textbox(label="SpO2 Level")
    
    with gr.Row():
        with gr.Column():
            generate = gr.Button("Analyze Video")
    
    generate.click(fn=analyze_video, inputs=[input_video], outputs=[bpm, breathing_rate, spo2, plot_output])

if __name__ == "__main__":
    demo.launch()
