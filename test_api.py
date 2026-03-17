import sys
import time
from openai import OpenAI

print("Starting API test...", flush=True)
client = OpenAI(
    api_key='sk-cp-9U5chKaokF9x4S_PCebURyA5N-92PyY5Rj5aDMEXZQ52MiMFGLRmyldZNNw-8QZTz3c1JsMFfuxDvzxQl6KTaybkGn_62XdJxAgMaZyoNXD-BNT8X8Joke4',
    base_url='https://api.minimax.io/v1',
    timeout=120.0,
)

print("Client created, sending request...", flush=True)
t0 = time.time()
try:
    r = client.chat.completions.create(
        model='MiniMax-M2.5',
        messages=[{'role': 'user', 'content': 'What is 2+2? Reply with just the number.'}],
        temperature=0,
        max_tokens=50,
    )
    elapsed = time.time() - t0
    print(f"SUCCESS in {elapsed:.1f}s", flush=True)
    print(f"Response: {r.choices[0].message.content}", flush=True)
except Exception as e:
    elapsed = time.time() - t0
    print(f"ERROR after {elapsed:.1f}s: {e}", flush=True)
    sys.exit(1)
