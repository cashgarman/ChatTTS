import hypercorn.asyncio
import hypercorn.config
from test_server import main
import asyncio
from hypercorn.asyncio import serve

if __name__ == "__main__":
	config = hypercorn.config.Config()
	config.bind = ["0.0.0.0:8080"]
	config.reload = True  # Enable hot-reloading

	print(f"config: {config}")
	print(f"main: {main}")
	asyncio.run(serve(main, config))  # Run the aiohttp app with hypercorn

