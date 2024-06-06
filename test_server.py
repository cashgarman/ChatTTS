import asyncio
import numpy as np
import aiohttp
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import AudioFrame
import os
import logging
from fractions import Fraction

def main(environ, start_response):
	print(f"environ: {environ}")
	print(f"start_response: {start_response}")
	logging.basicConfig(level=logging.DEBUG)

	SAMPLE_RATE = 48000
	DURATION = 10  # seconds
	FREQUENCY = 440.0  # Hz

	# Generate a sine wave
	t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
	audio_data = (0.5 * np.sin(2 * np.pi * FREQUENCY * t)).astype(np.float32)

	# WebRTC configuration
	pcs = set()

	class SineWaveAudioStreamTrack(MediaStreamTrack):
		kind = "audio"

		def __init__(self):
			super().__init__()
			self.audio_data = audio_data
			self.sample_rate = SAMPLE_RATE
			self.samples_per_frame = int(self.sample_rate / 100)
			self.frame_index = 0

		async def recv(self):
			start = self.frame_index * self.samples_per_frame
			end = start + self.samples_per_frame
			frame_data = self.audio_data[start:end]

			if end >= len(self.audio_data):
				self.frame_index = 0
			else:
				self.frame_index += 1

			frame = AudioFrame.from_ndarray(frame_data, format='float32', sample_rate=self.sample_rate)
			frame.time_base = Fraction(1, self.sample_rate)
			return frame

	async def offer(request):
		try:
			params = await request.json()
			logging.debug(f"Received offer with params: {params}")
			offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
			pc = RTCPeerConnection()
			pcs.add(pc)

			@pc.on('iceconnectionstatechange')
			async def on_iceconnectionstatechange():
				logging.info(f'ICE connection state: {pc.iceConnectionState}')
				if pc.iceConnectionState == 'failed':
					await pc.close()
					pcs.discard(pc)

			@pc.on('datachannel')
			async def on_datachannel(channel):
				logging.info(f'Data channel: {channel.label}')

			@pc.on('track')
			async def on_track(track):
				logging.info(f'Track received: {track.kind}')

			audio_track = SineWaveAudioStreamTrack()
			pc.addTrack(audio_track)

			await pc.setRemoteDescription(offer)
			answer = await pc.createAnswer()
			await pc.setLocalDescription(answer)

			return web.json_response({
				'sdp': pc.localDescription.sdp,
				'type': pc.localDescription.type
			})
		except Exception as e:
			logging.error(f"Error in offer handler: {e}")
			return web.Response(status=500, text=str(e))

	async def on_shutdown(app):
		coros = [pc.close() for pc in pcs]
		await asyncio.gather(*coros)
		pcs.clear()

	app = web.Application()
	app.on_shutdown.append(on_shutdown)
	app.router.add_post('/offer', offer)

	# Serve static files
	static_path = os.path.abspath(os.path.dirname(__file__))
	app.router.add_static('/', static_path)