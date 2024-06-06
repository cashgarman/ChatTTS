# Set the logging level to debug
import logging
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

import random
import argparse
import time

import torch
torch._dynamo.config.cache_size_limit = 32
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('medium')

import gradio as gr
import numpy as np
import librosa
import ChatTTS

import cProfile

# so if an A [uv_break] G [uv_break] I [uv_break] is a chat bot that is pretty much an expert in all fields, [uv_break] that ship has sailed... If an A [uv_break] G [uv_break] I [uv_break] is something more than that, it definitely lacks anything close to a legal definition. [laugh]
# I find it really funny that the AI is holding back a laugh about the fact that the AGI ship has sailed. [laugh] Even if it is from the script you gave it. [laugh]

#  Good Seeds:
# 25124167
# 11988514
# 21766748

shortcut_js = """
<script>
//function shuffleAudio()
//{
//	// Randomize the audio seed
//	document.getElementById("audio_seed_input").value = Math.floor(Math.random() * 100000000);
//	document.getElementById("text_seed_input").value = Math.floor(Math.random() * 100000000);
//
//	// Click the generate button
//	document.getElementById("generate_button").click();
//
//	// Repeat the process
//	setTimeout(shuffleAudio, 10000);
//}

function shortcuts(e) {
	console.log("Key pressed");
    var event = document.all ? window.event : e;
	if (e.ctrlKey && e.key === 'Enter') {
		console.log("Generate button clicked");
		//shuffleAudio();
	}
}

document.addEventListener('keydown', shortcuts, false);
console.log("Shortcuts enabled");
</script>
"""

def generate_seed():
	new_seed = random.randint(1, 100000000)
	return {
		"__type__": "update",
		"value": new_seed
		}

def generate_audio(text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag, refine_text_prompt):
	
	# Log if CUDA is available
	if torch.cuda.is_available():
		print("CUDA is available")
	else:
		print("CUDA is not available")

	start_time = time.time()
	torch.manual_seed(audio_seed_input)
	rand_spk = chat.sample_random_speaker()
	params_infer_code = {
		'spk_emb': rand_spk, 
		'temperature': temperature,
		'top_P': top_P,
		'top_K': top_K,
		}
	params_refine_text = {'prompt': refine_text_prompt}
	
	torch.manual_seed(text_seed_input)

	if refine_text_flag:
		text = chat.infer(text, 
						skip_refine_text=False,
						refine_text_only=True,
						params_refine_text=params_refine_text,
						params_infer_code=params_infer_code
						)
	
	wav = chat.infer(text, 
					skip_refine_text=True, 
					params_refine_text=params_refine_text, 
					params_infer_code=params_infer_code
					)
	
	audio_data = np.array(wav[0]).flatten()
	sample_rate = 24000
	text_data = text[0] if isinstance(text, list) else text

	# Trim any silence from the start and end of the audio data
	timer = time.time()
	trimmed_audio, _ = librosa.effects.trim(audio_data, top_db=40)
	print(f"Trimming time: {time.time() - timer}")

	end_time = time.time()
	generation_time = end_time - start_time

	return [(sample_rate, trimmed_audio), text_data, generation_time]


def main():

	with gr.Blocks(head=shortcut_js) as demo:
		gr.Markdown("# ChatTTS Webui")
		gr.Markdown("ChatTTS Model: [2noise/ChatTTS](https://github.com/2noise/ChatTTS)")

		default_text = "The quick brown fox jumps over the lazy dog."
		text_input = gr.Textbox(label="Input Text", lines=4, placeholder="Please Input Text...", value=default_text)
		refine_text_prompt = gr.Textbox(label="Refine Text Prompt", value="[oral_2][laugh_0][break_6]", placeholder="Enter refine text prompt...")

		initial_values = {
			"refine_text": False,
			"temperature": 0.3,
			"top_P": 0.7,
			"top_K": 20,
			"audio_seed": 2,
			"text_seed": 42,
			"auto_play": True
		}

		with gr.Row():
			refine_text_checkbox = gr.Checkbox(label="Refine text", value=initial_values["refine_text"])
			temperature_slider = gr.Slider(minimum=0.00001, maximum=1.0, step=0.00001, value=initial_values["temperature"], label="Audio temperature")
			top_p_slider = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=initial_values["top_P"], label="top_P")
			top_k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=initial_values["top_K"], label="top_K")
			auto_play = gr.Checkbox(label="Auto Play", value=initial_values["auto_play"])

		with gr.Row():
			audio_seed_input = gr.Number(value=initial_values["audio_seed"], label="Audio Seed", elem_id="audio_seed_input")
			generate_audio_seed = gr.Button("\U0001F3B2")
			text_seed_input = gr.Number(value=initial_values["text_seed"], label="Text Seed", elem_id="text_seed_input")
			generate_text_seed = gr.Button("\U0001F3B2")

		generate_button = gr.Button("Generate", elem_id="generate_button")
		restore_defaults_button = gr.Button("Restore Defaults")
		
		text_output = gr.Textbox(label="Output Text", interactive=False)
		audio_output = gr.Audio(label="Output Audio", autoplay=auto_play)
		generation_time_output = gr.Textbox(label="Generation Time (s)", interactive=False)

		generate_audio_seed.click(generate_seed, 
								inputs=[], 
								outputs=audio_seed_input)

		generate_text_seed.click(generate_seed, 
								inputs=[], 
								outputs=text_seed_input)
		
		generate_button.click(generate_audio, 
						inputs=[text_input, temperature_slider, top_p_slider, top_k_slider, audio_seed_input, text_seed_input, refine_text_checkbox, refine_text_prompt], 
						outputs=[audio_output, text_output, generation_time_output])

		def restore_defaults():
			return (initial_values["refine_text"], initial_values["temperature"], initial_values["top_P"], initial_values["top_K"], initial_values["audio_seed"], initial_values["text_seed"], "[oral_2][laugh_0][break_6]")

		restore_defaults_button.click(restore_defaults, 
									inputs=[], 
									outputs=[refine_text_checkbox, temperature_slider, top_p_slider, top_k_slider, audio_seed_input, text_seed_input, refine_text_prompt])

	parser = argparse.ArgumentParser(description='ChatTTS demo Launch')
	parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
	parser.add_argument('--server_port', type=int, default=8080, help='Server port')
	parser.add_argument('--local_path', type=str, default=None, help='the local_path if need')
	args = parser.parse_args()

	print("Creating ChatTTS...")
	global chat
	timer = time.time()
	chat = ChatTTS.Chat()
	print(f"Init time: {time.time() - timer}")

	print("Loading model...")
	timer = time.time()
	if args.local_path == None:
		chat.load_models()
	else:
		print('local model path:', args.local_path)
		chat.load_models('local', local_path=args.local_path)
	print(f"Loading time: {time.time() - timer}")

	demo.launch(server_name=args.server_name, server_port=args.server_port, inbrowser=False)

if __name__ == '__main__':
	# cProfile.run('main()', '~/ChatTTS/profile.prof')
	main()