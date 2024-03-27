import asyncio
import json
import os
import sys
import traceback
import json
import asyncio
import aiofiles

from AI.AIProcessor import faintly2
max_http_connections_semaphore = None

# from web.web_scan import pass_article
callbacks = {}
callbacks_active =[]
is_ready_to_be_done = False

def all_done():
    global is_ready_to_be_done
    is_ready_to_be_done = True
items = []
l = []
item_names = []
lock = asyncio.Lock()
async def task_init_run_callbacks(task:str, data,stage_min: int= 0):
        async with lock:
            while len(items) > 1 and len(l) >3:
                await asyncio.sleep(10)
            items.append({"task": task, "data": data, "stage_min": stage_min})

async def run_callbacks(task:str, data,stage_min: int= 0):
        async with lock:
            while len(items) > 50 and len(l) >50:
                await asyncio.sleep(0.25)
            items.append({"task": task, "data": data, "stage_min": stage_min})

async def pop_task_list_from_file():
    # async with lock:
        task = None
        while len(items) == 0:
            await asyncio.sleep(1)
        task = items.pop(0)
        return task

async def helper_setup():
    global max_http_connections_semaphore
    max_http_connections_semaphore = asyncio.Semaphore(10)

def get_max_http_connections_semaphore():
    global max_http_connections_semaphore
    return max_http_connections_semaphore

def add_callback(callback, task: str, isAsyncTask: bool, stage: int= 0):
    if task not in callbacks:
        callbacks[task] = {}
    if stage not in callbacks[task]:
        callbacks[task][stage] = {
            "asyncTask": [],
            "syncTask": []
        }
    if isAsyncTask:
        callbacks[task][stage]["asyncTask"].append(callback)
    else:
        callbacks[task][stage]["syncTask"].append(callback)


# async def run_callbacks(task:str, data,stage_min: int= 0):
#     await add_task_to_file(task,data, stage_min)
#     # callbacks_active.append({"task":task,"data":data,"stage_min":stage_min})

async def do_callbacks():
    global l
    global item_names
    print("do_callbacks")
    while True:
        try:
            active =  await pop_task_list_from_file()
            task = active["task"] 
            stage_min = active["stage_min"]
            data = active["data"]
            item_names.append(task)
            l.append(asyncio.create_task(_do_callbacks(task, data, stage_min)))
            if len(l) > 100:
                for ll in l:
                    try:
                        await ll
                    except:
                        pass
                l = []
                item_names = []
        except Exception as e:
            print("do_callbacks vxcxcv", e)

async def _do_callbacks(task:str, data, stage_min):
        try:
            if task in callbacks.keys():
                for stage in sorted(list(callbacks[task].keys())):
                    if stage <= stage_min:
                        continue
                    if stage in callbacks[task] and isinstance(callbacks[task][stage], dict):
                        for callback in callbacks[task][stage]["syncTask"]:
                            try:
                                data_out = callback(data)
                                if data_out is not None:
                                    data = data_out
                            except Exception as e:
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                fname = traceback.extract_tb(exc_tb)[-1][2]
                                if data is not None  :
                                    pass
                                    print(exc_type, fname, exc_tb.tb_lineno,task, None, stage_min)
                                    print("2.1 run_callbacks", e,task , None,callback)
                                else:
                                    pass
                                    print(exc_type, fname, exc_tb.tb_lineno,task, data.keys(), stage_min)
                                    print("2.2 run_callbacks", e,task , data.keys(),callback)
                        for callback in callbacks[task][stage]["asyncTask"]:
                            try:
                                data_out = await callback(data)
                                if data_out is not  None:
                                    data = data_out
                            except Exception as e:
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                fname = traceback.extract_tb(exc_tb)[-1][2]
                                if data is None:
                                    pass
                                    print(exc_type, fname, exc_tb.tb_lineno,task, None, stage_min)
                                    print("3.1 run_callbacks", e,task , None,callback)
                                else:
                                    pass
                                    print(exc_type, fname, exc_tb.tb_lineno,task, data.keys(), stage_min)
                                    print("3.2 run_callbacks", e,task , data.keys(),callback)
            # await faintly(task, data)
            await faintly2(task, data)
        except Exception as e:
            # get line number of error
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = traceback.extract_tb(exc_tb)[-1][2]
            if data is None:
                print(exc_type, fname, exc_tb.tb_lineno,task, data, stage_min)
                print("4 run_callbacks", e)
            else:
                print(exc_type, fname, exc_tb.tb_lineno,task, data.keys(), stage_min)
                print("4 run_callbacks", e)

    
    