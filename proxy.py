from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import time
import asyncio
import threading

CLASSIFICATION_SERVER_URL = "http://localhost:8001/classify"
MAX_BATCH_SIZE = 5
MAX_RETRIES = 3
TIMEOUT = 3
BATCH_TIMEOUT = 0.05  # 50 milliseconds

request_queue = asyncio.Queue()
response_dict = {}
response_lock = threading.Lock()

client_a_time = 0
client_b_time = 0
total_execution_time = 0
metrics_lock = threading.Lock()

app = FastAPI(
    title="Classification Proxy",
    description="Proxy server that handles rate limiting and retries for the code classification service"
)

class ProxyRequest(BaseModel):
    sequence: str
    client_type: str

class ProxyResponse(BaseModel):
    result: str

async def process_batch_task():
    """Continuously process batches with timeout support."""
    while True:
        batch = []
        first_req = await request_queue.get()
        batch.append(first_req)

        start_time = time.time()

        while len(batch) < MAX_BATCH_SIZE and (time.time() - start_time) < BATCH_TIMEOUT:
            try:
                remaining_time = BATCH_TIMEOUT - (time.time() - start_time)
                req = await asyncio.wait_for(request_queue.get(), timeout=remaining_time)
                batch.append(req)
            except asyncio.TimeoutError:
                break

        await send_batch_to_classification_server(batch)

async def send_batch_to_classification_server(batch):
    """Send batch to classification server with retry logic."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                sequences = [req.sequence for req in batch]
                response = await client.post(CLASSIFICATION_SERVER_URL, json={"sequences": sequences})

                if response.status_code == 200:
                    data = response.json()
                    for i, req in enumerate(batch):
                        with response_lock:
                            response_dict[req.sequence] = data["results"][i]
                        # Metrics
                        with metrics_lock:
                            elapsed = time.time() - req.start_time
                            if req.client_type == "Client A":
                                global client_a_time
                                client_a_time += elapsed
                            else:
                                global client_b_time
                                client_b_time += elapsed
                            global total_execution_time
                            total_execution_time += elapsed
                    break
                elif response.status_code == 429:
                    retries += 1
                    await asyncio.sleep(0.1)
                else:
                    response.raise_for_status()
            except Exception:
                retries += 1
                await asyncio.sleep(0.1)
                if retries == MAX_RETRIES:
                    for req in batch:
                        with response_lock:
                            response_dict[req.sequence] = "error"

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_batch_task())

@app.post("/proxy_classify")
async def proxy_classify(req: ProxyRequest):
    req.start_time = time.time()
    await request_queue.put(req)

    while True:
        with response_lock:
            if req.sequence in response_dict:
                result = response_dict.pop(req.sequence)
                return ProxyResponse(result=result)
        await asyncio.sleep(0.01)

@app.get("/results")
def get_results():
    with metrics_lock:
        return {
            "Client A Total Time": round(client_a_time, 2),
            "Client B Total Time": round(client_b_time, 2),
            "Total Execution Time": round(total_execution_time, 2)
        }