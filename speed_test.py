import ChatTTS
from IPython.display import Audio
import torchaudio
import torch
import numpy as np

# Set the cache size limit and suppress errors
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('medium')

import time
import matplotlib.pyplot as plt
from pydub import AudioSegment
import random

# Set the style and colors
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = 'darkgreen'
plt.rcParams['figure.facecolor'] = 'darkgreen'
plt.rcParams['text.color'] = 'lime'
plt.rcParams['axes.labelcolor'] = 'lime'
plt.rcParams['xtick.color'] = 'lime'
plt.rcParams['ytick.color'] = 'lime'
plt.rcParams['legend.edgecolor'] = 'lime'
plt.rcParams['legend.facecolor'] = 'darkgreen'

# Get the start time
initial_start_time = time.time()

# Initialize ChatTTS
start_time = time.time()
chat = ChatTTS.Chat()
init_time = time.time() - start_time
print(f"ChatTTS initialization time: {init_time:.1f}s")

# Load models
start_time = time.time()
chat.load_models(compile=True) # Set to True for better performance
model_load_compile_time = time.time() - start_time
print(f"Model and compiling time: {model_load_compile_time:.1f}s")

# Initialize lists to store the times
inference_times = []
saving_times = []
audio_lengths = []

# Set the seed for reproducibility
seed = 2

# Set the prompts
prompts = [
	"A swift azure fox leaps over three dormant hounds.[uv_break]",
	"Two lazy dogs watch a quick brown fox sprinting.",
	"Jumping jubilantly, the brown fox outpaces the old dog.[uv_break]",
	"Under the moonlight, a fox quietly tiptoes past the sleeping dog.",
	"Energetic foxes race while the dog daydreams.[uv_break]",
	"The fox hops over one, then two, then three lazy dogs.",
	"A fox, quick and bright, vaults over a dog deep in slumber.[uv_break]",
	"Silently, a clever fox outsmarts the slow, drowsy dog.",
	"The dog sleeps as the fox creeps swiftly over.[uv_break]",
	"Quickly, the fox jumps; the dog, lazy, barely notices."
]

for i in range(10):
	# Get the next prompt
	text = prompts[i]

	# Every other iteration, double up the text with another random prompt to help measure generation speed for longer prompts
	if i % 2 == 0:
		text = f"{text} {prompts[random.randint(0, len(prompts)-1)]}"

	# Reset the seed for reproducibility
	torch.manual_seed(seed)

	# Set the parameters for inference
	params_infer_code = {
		'spk_emb': chat.sample_random_speaker(), 
		'temperature': 0.3,
		'top_P': 0.7,
		'top_K': 20,
	}

	# Set the parameters for refinement
	params_refine_text = {'prompt': text}

	# Infer audio from text
	start_time = time.time()
	wavs = chat.infer(text, 
					skip_refine_text=True, 
					params_refine_text=params_refine_text, 
					params_infer_code=params_infer_code
					)
	inference_time = time.time() - start_time
	inference_times.append((inference_time, seed))
	print(f"Inference time {i}: {inference_time:.1f}s")

	# Save output to WAV file
	start_time = time.time()
	torchaudio.save(f"output{i}_{seed}.wav", torch.from_numpy(wavs[0]), 24000)
	saving_time = time.time() - start_time
	saving_times.append((saving_time, seed))
	print(f"Audio saving time {i}: {saving_time:.1f}s")

	# Calculate the audio duration
	start_time = time.time()
	audio_segment = AudioSegment(wavs[0].tobytes(), frame_rate=24000, sample_width=wavs[0].dtype.itemsize, channels=1)
	audio_duration = len(audio_segment) / 1000.0  # Duration in seconds
	audio_lengths.append(audio_duration)
	print(f"Audio duration {i}: {audio_duration:.1f}s (calculated in {time.time() - start_time:.1f}s)")

# Plotting the times
iterations = list(range(10))
plt.figure(figsize=(10, 5))

# Create the bar plot for inference times
speed_bars = plt.bar(iterations, [x[0] for x in inference_times], width=0.4, label='Inference Time', align='center')
plt.xlabel('Iteration')
plt.ylabel('Time (seconds)')
plt.title('Inference Times per Iteration')
plt.legend()

# Add labels above the bars
for i, bar in enumerate(speed_bars):
	yval = bar.get_height()
	plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'Inference: {yval:.1f}s\nLength: {audio_lengths[i]:.1f}s\nSeed: {inference_times[i][1]}', 
            ha='center', va='bottom', fontsize=6)

# Add configuration details as text on the plot
precision = torch.get_float32_matmul_precision()
config_text = f"Cache Size Limit: {torch._dynamo.config.cache_size_limit}\nMatmul Precision: {precision}"
plt.gcf().text(0.02, 0.95, config_text, fontsize=10, verticalalignment='top')

# Save the plot as a PNG file
plt.savefig(f'profile_graphs/timing_graph_{time.strftime("%Y-%m-%d %H:%M:%S")}.png')

# Print the total time
print(f"Total time: {time.time() - initial_start_time:.1f}s")