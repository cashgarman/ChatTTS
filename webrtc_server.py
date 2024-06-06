import logging
import random
import time
import torch
import numpy as np
import asyncio
import json

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay

import ChatTTS

torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.DEBUG)

# Initialize ChatTTS model
chat = ChatTTS.Chat()
chat.load_models()

relay = MediaRelay()

class AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag, refine_text_prompt):
        super().__init__()
        self.text = text
        self.temperature = temperature
        self.top_P = top_P
        self.top_K = top_K
        self.audio_seed_input = audio_seed_input
        self.text_seed_input = text_seed_input
        self.refine_text_flag = refine_text_flag
        self.refine_text_prompt = refine_text_prompt

    async def recv(self):
        torch.manual_seed(self.audio_seed_input)
        rand_spk = chat.sample_random_speaker()
        params_infer_code = {
            'spk_emb': rand_spk, 
            'temperature': self.temperature,
            'top_P': self.top_P,
            'top_K': self.top_K,
        }
        params_refine_text = {'prompt': self.refine_text_prompt}

        torch.manual_seed(self.text_seed_input)

        if self.refine_text_flag:
            text = chat.infer(self.text, 
                            skip_refine_text=False,
                            refine_text_only=True,
                            params_refine_text=params_refine_text,
                            params_infer_code=params_infer_code
                            )
        
        wav = chat.infer(self.text, 
                        skip_refine_text=True, 
                        params_refine_text=params_refine_text, 
                        params_infer_code=params_infer_code
                        )
        
        audio_data = np.array(wav[0]).flatten()
        sample_rate = 24000

        # Convert numpy array to bytes
        audio_bytes = audio_data.tobytes()

        # Send the audio bytes
        frame = MediaStreamTrack.AudioFrame(audio_bytes, sample_rate)
        return frame

async def index(request):
    content = open('index.html', 'r').read()
    return web.Response(content_type='text/html', text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on('datachannel')
    async def on_datachannel(channel):
        @channel.on('message')
        async def on_message(message):
            data = json.loads(message)
            text = data['text']
            temperature = data['temperature']
            top_P = data['top_P']
            top_K = data['top_K']
            audio_seed_input = data['audio_seed_input']
            text_seed_input = data['text_seed_input']
            refine_text_flag = data['refine_text_flag']
            refine_text_prompt = data['refine_text_prompt']

            audio_track = AudioTrack(
                text, temperature, top_P, top_K, audio_seed_input, text_seed_input, refine_text_flag, refine_text_prompt
            )
            pc.addTrack(audio_track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})
    )

pcs = set()

app = web.Application()
app.router.add_get('/', index)
app.router.add_post('/offer', offer)

web.run_app(app, port=8080)