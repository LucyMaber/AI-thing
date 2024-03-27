import asyncio

tasks = []
count = None
async def task_init():
    global tasks
    global count
    tasks = []
    count = asyncio.Semaphore(100)

async def add_task(task):
    global count
    await count.acquire()
    tasks.append(task)
    return tasks

async def run_tasks():
    print("Running tasks")
    global tasks
    global count

    async def run_task(task):
        print("Running task",task)
        await task()
    while True:
        await asyncio.gather(*[asyncio.create_task(run_task(task)) for task in tasks])