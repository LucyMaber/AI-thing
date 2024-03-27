import asyncio
from bs4 import BeautifulSoup
import aiohttp
import urllib.parse
import re

import httpx
timeout = 60
strps = 20


async def search_archive(query):
    urls = []
    offset = 0
    total = 1
    # Properly encode the query

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Accept": "*/*",
        "Accept-Language": "en-GB,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://archive.li/",
        "Sec-Fetch-Dest": "script",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "TE": "trailers",
    }
    while offset < total:
        search_url = f'https://archive.li/offset={offset}/' + query
        for i in range(10):
            try:
                async with aiohttp.AsyncClient(headers=headers) as client:
                    response = await client.get(search_url)
                    # You can handle the response here
                    # For example, you can get the HTML content like this:

                    if response.status_code != 200:
                        print("response",response.headers)
                        await asyncio.sleep(60*5)
                        continue
                    html_content = response.text
                    soup = BeautifulSoup(html_content, 'html.parser')
                    pager = soup.select("[id=pager]")
                    pager = pager[0].text
                    m = re.search(
                        "([0-9]*)\.\.([0-9]*) of ([0-9]*) urls", pager)
                    start = int(m.group(1))
                    end = int(m.group(2))
                    offset = offset + 20
                    total = int(m.group(3))
                    if offset <= total:
                        await asyncio.sleep(timeout)
                    search_results = soup.select('[id*=row] a[href]')
                    for result in search_results:
                        urls.append(result['href'])
            except Exception as e:
                print(e, "retrying", i)
                continue
    for url in urls:
        async with httpx.AsyncClient(headers=headers) as client:
            response = await client.get(url)
            await asyncio.sleep(timeout)

# # Replace 'your_query' with the query you want to search
# query = 'reddit.com/r/*'

# loop = asyncio.get_event_loop()
# loop.run_until_complete(search_archive(query))
