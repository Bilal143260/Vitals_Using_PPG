import gradio as gr
import numpy as np
import imageio
import matplotlib.pyplot as plt

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

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(16,9))
    plt.plot(x_filt, colors['red_filt'], color='#fc4f30')
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Pixel Color')
    plt.title('Filtered Red Channel Pixel Data')
    fig2 = plt.gcf()
    # fig2.savefig('filtered.png', dpi=200)

    return heartrate, fig2



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_video = gr.PlayableVideo(format="mp4", interactive=True, 
                                           autoplay=True, width="600px", height="400px")
        with gr.Column():
            # graphs = gr.Gallery(preview=True, object_fit="scale-down", interactive=False)
            plot_output = gr.Plot()
            bpm = gr.Textbox(label="BMP")
    
    with gr.Row():
        with gr.Column():
            generate = gr.Button()
        
    
    generate.click(fn=find_bpm,inputs=[input_video],outputs=[bpm,plot_output])

if __name__ == "__main__":
    demo.launch()