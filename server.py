import logging
import random
import time

import torch
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import numpy as np
import argparse
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import AudioFrame, MediaStreamError
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import ChatTTS
import json
from fractions import Fraction

# Set the logging level to debug
logging.basicConfig(level=logging.DEBUG)

# Create logger
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(ch)

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Initialize the ChatTTS model globally
logger.debug("Loading ChatTTS model...")
chat = ChatTTS.Chat()
chat.load_models()
logger.debug("ChatTTS model loaded.")

def generate_audio(text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag, refine_text_prompt):
	logger.debug(f"Generating audio for text: {text}")
	start_time = time.time()

	torch.manual_seed(audio_seed_input)
	rand_spk = chat.sample_random_speaker()
	logger.debug(f"Random speaker sampled: {rand_spk}")
	params_infer_code = {
		'spk_emb': rand_spk, 
		'temperature': temperature,
		'top_P': top_P,
		'top_K': top_K,
	}
	params_refine_text = {'prompt': refine_text_prompt}
	
	torch.manual_seed(text_seed_input)

	if refine_text_flag:
		logger.debug("Refining text...")
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

	end_time = time.time()
	generation_time = end_time - start_time
	logger.debug(f"Audio generated in {generation_time} seconds.")

	return (sample_rate, audio_data), text_data, generation_time

@app.post("/generate")
async def generate(request: Request):
	data = await request.json()
	logger.debug(f"Received request data: {data}")
	text = data.get("text", "The quick brown fox jumps over the lazy dog.")
	temperature = data.get("temperature", 0.3)
	top_P = data.get("top_P", 0.7)
	top_K = data.get("top_K", 20)
	audio_seed_input = data.get("audio_seed", 2)
	text_seed_input = data.get("text_seed", 42)
	refine_text_flag = data.get("refine_text", False)
	refine_text_prompt = data.get("refine_text_prompt", "[oral_2][laugh_0][break_6]")

	# Log all the parameters
	logger.debug(f"Text: {text}")
	logger.debug(f"Temperature: {temperature}")
	logger.debug(f"Top P: {top_P}")
	logger.debug(f"Top K: {top_K}")
	logger.debug(f"Audio seed input: {audio_seed_input}")
	logger.debug(f"Text seed input: {text_seed_input}")
	logger.debug(f"Refine text flag: {refine_text_flag}")
	logger.debug(f"Refine text prompt: {refine_text_prompt}")

	logger.debug("Generating audio...")
	(sample_rate, audio_data), text_data, generation_time = generate_audio(
		text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag, refine_text_prompt
	)
	logger.debug("Audio generated.")

	response = {
		"sample_rate": sample_rate,
		"audio_data": audio_data.tolist(),
		"text_data": text_data,
		"generation_time": generation_time
	}
	logger.debug(f"Response data: {response}")
	return JSONResponse(response)

class AudioStreamTrack(MediaStreamTrack):
	kind = "audio"

	def __init__(self, audio_data, sample_rate):
		super().__init__()
		self.audio_data = audio_data
		self.sample_rate = sample_rate
		self.frame_index = 0
		self.samples_per_frame = sample_rate // 100  # Assuming 10ms frames
		logger.debug("AudioStreamTrack initialized.")

	async def recv(self):
		logger.debug(f"Receiving audio frame: {self.frame_index}")
		if self.frame_index >= len(self.audio_data):
			raise MediaStreamError

		frame_data = self.audio_data[self.frame_index:self.frame_index + self.samples_per_frame]
		self.frame_index += self.samples_per_frame

		frame = AudioFrame.from_ndarray(frame_data, format="s16", sample_rate=self.sample_rate)
		frame.time_base = Fraction(1, self.sample_rate)
		return frame

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
	await websocket.accept()
	logger.debug("WebSocket connection accepted.")
	pc = RTCPeerConnection()

	@pc.on("datachannel")
	def on_datachannel(channel):
		@channel.on("message")
		async def on_message(message):
			logger.debug(f"Received message on data channel: {message}")
			data = json.loads(message)
			text = data.get("text", "The quick brown fox jumps over the lazy dog.")
			temperature = float(data.get("temperature", 0.3))
			top_P = float(data.get("top_P", 0.7))
			top_K = int(data.get("top_K", 20))
			audio_seed_input = int(data.get("audio_seed", 2))
			text_seed_input = int(data.get("text_seed", 42))
			refine_text_flag = data.get("refine_text", False)
			refine_text_prompt = data.get("refine_text_prompt", "[oral_2][laugh_0][break_6]")

			# Log all the parameters
			logger.debug(f"Text: {text}")
			logger.debug(f"Temperature: {temperature}")
			logger.debug(f"Top P: {top_P}")
			logger.debug(f"Top K: {top_K}")
			logger.debug(f"Audio seed input: {audio_seed_input}")
			logger.debug(f"Text seed input: {text_seed_input}")
			logger.debug(f"Refine text flag: {refine_text_flag}")
			logger.debug(f"Refine text prompt: {refine_text_prompt}")

			# Send an acknowledgement to the client with all the parameters
			await websocket.send_json({
				"ack": "ack",
				"text": text,
				"temperature": temperature,
				"top_P": top_P,
				"top_K": top_K,
				"audio_seed": audio_seed_input,
				"text_seed": text_seed_input,
				"refine_text": refine_text_flag,
				"refine_text_prompt": refine_text_prompt
			})

			(sample_rate, audio_data), text_data, generation_time = generate_audio(
				text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag, refine_text_prompt
			)

			audio_track = AudioStreamTrack(audio_data, sample_rate)
			pc.addTrack(audio_track)
			logger.debug("Audio track added to peer connection.")

			# Play the audio
			logger.debug("Playing audio...")
			audio_track.play()
			logger.debug("Audio playing.")

	@pc.on("icecandidate")
	async def on_icecandidate(candidate):
		logger.debug(f"ICE candidate: {candidate}")
		await websocket.send_json({"candidate": candidate})

	offer = await websocket.receive_json()
	logger.debug(f"Received offer: {offer}")
	await pc.setRemoteDescription(RTCSessionDescription(offer["sdp"], offer["type"]))
	answer = await pc.createAnswer()
	await pc.setLocalDescription(answer)
	await websocket.send_json({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
	logger.debug("Sent answer to WebSocket.")

	try:
		while True:
			await websocket.receive_text()
	except WebSocketDisconnect:
		logger.debug("WebSocket disconnected.")
	finally:
		await pc.close()
		logger.debug("Peer connection closed.")

if __name__ == "__main__":
	import uvicorn
	parser = argparse.ArgumentParser(description='ChatTTS API Server')
	parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
	parser.add_argument('--port', type=int, default=8080, help='Port number')
	args = parser.parse_args()
	logger.debug(f"Starting server at {args.host}:{args.port}")
	uvicorn.run(app, host=args.host, port=args.port, log_config=None)
