import asyncio
import datetime
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
import aiohttp
from datetime import datetime

import httpx
import waybackpy
import utility.helper as helper
class Wayback_Viewer:
    def __init__(self, url, mine, when: datetime) -> None:
        self.url = url
        self.mine = mine
        self.when = when

    def get_url(self):
        return self.url

    def get_mine(self):
        return self.mine

    async def get_body(self):
        # this is a windows computer ;)
        user_agent = "Mozilla/5.0 (Windows NT 5.1; rv:40.0) Gecko/20100101 Firefox/40.0"
        availability_api = waybackpy.Url(self.url, user_agent)
        near = availability_api.near(year=self.when.year, month=self.when.month,
                                     day=self.when.day, hour=self.when.hour, minute=self.when.hour)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'
        }
        for i in range(10):
            try:
                async with aiohttp.AsyncClient() as session:
                    async with session.get(str(near), headers=headers) as response:
                        text = await response.content.read()
                    return text
            except Exception as e:
                print("wayback error: ", e)
                print("retrying count: "+str(i), "url:", near)
                await asyncio.sleep(10)
        return None


async def get_wayback_cdx(url):
    data = None
    for i in range(10):
        try:
            async with httpx.AsyncClient() as session:
                response = await session.get(url+"&output=json")
                data = await response.json()
            break
        except Exception as e:
            print("wayback error: ", e)
            print("retrying count: "+str(i), "url:", url)
            await asyncio.sleep(10)
    if data is None:
        return []
    result = []  # Create an empty list to store the dictionaries
    try:
        if len(data) != 0:
            keys = data[0]  # Extract the keys from the first row
            for row in data[1:]:  # Iterate through each row starting from the second row
                # Create a dictionary using keys and row
                dict_row = dict(zip(keys, row))
                # Append the dictionary to the result list
                result.append(dict_row)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("get_wayback_cdx error:", e)
        result = []
    return result


def add_to_dict(data_list, item_name, dt_object):
    if item_name not in data_list.keys():
        data_list[item_name] = {
            "start": dt_object,
            "end": dt_object,
        }
    else:
        if data_list[item_name]["start"] < dt_object:
            data_list[item_name]["start"] = dt_object
        if data_list[item_name]["end"] > dt_object:
            data_list[item_name]["end"] = dt_object
    return data_list


async def get_decode(data_list):
    for i in data_list:
        timestamp = i['timestamp']
        date_format = "%Y%m%d%H%M%S"
        dt_object = datetime.strptime(timestamp, date_format)
        data = Wayback_Viewer(i['original'], i['mimetype'], dt_object)
        await helper_cache_url(i['original'], dt_object, data,
                               "weyback", location=i['digest'], mime=i['mimetype'])


# async def get_url(url):
#     url = f"https://web.archive.org/cdx/search/cdx?url={url}/*&filter=statuscode:200&filter=mimetype:application/json"
#     data_list = await get_wayback_cdx(url)
#     print(data_list)
# print(get_url)
# # helper.add_callbacks_async_web_active(get_url)
