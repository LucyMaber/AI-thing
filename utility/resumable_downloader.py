import aiohttp
import aiofiles
import os
import asyncio
import json

from utility.helper import get_max_http_connections_semaphore

class ResumableDownloader:
    def __init__(self, url, download_path):
        self.url = url
        # os get cwd
        p_cwd = os.getcwd()
        self.download_path = os.path.join(p_cwd, download_path)
        self.headers_path = os.path.join(p_cwd, download_path) + '.headers'

    async def download(self):
        # check if file headers exists
        old_headers  ={}
        file_size = 0
        if os.path.isfile(self.headers_path):
            # if exists, load headers
            old_headers = await self.load_headers()
        # check if we allrady have the file
        if os.path.isfile(self.download_path):
            # get the length of the file
            file_size = os.path.getsize(self.download_path)
        await self.__download__(have_size=file_size, old_headers=old_headers)

    async def __download__(self, have_size = 0, old_headers={}):
        headers_out = self.partial_content(old_headers, have_size)
        async with aiohttp.ClientSession() as session:
                async with session.head(self.url) as response:
                    new_headers = response.headers
                    await self.save_headers(new_headers)
                    same = self.header_same_version(new_headers, old_headers)
        if same == "same":
            if 'Content-Length' in response.headers.keys():
                content_length = response.headers.get('Content-Length')
                if content_length == have_size:
                    return
            await self.same_version_download(headers_out)
        elif same == "different":
            await self.different_version_download()


    async def same_version_download(self, headers_out):
        #  get file size
        print("same_version_download")
        file_size = os.path.getsize(self.download_path)
        count_seek = 0
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url, headers=headers_out) as response:
                async with aiofiles.open(self.download_path, mode='wb+') as fw:
                    async for data, _ in response.content.iter_chunks():
                        data_size = len(data)
                        count_seek += data_size
                        if count_seek < file_size:
                            continue
                        await fw.write(data)

    async def different_version_download(self):
        print("different_version_download")
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as response:
                async with aiofiles.open(self.download_path, mode='wb') as fw:
                    async for data, _ in response.content.iter_chunks():
                        fw.write(data)
                
    async def save_headers(self, headers):
        output = {}
        for key, value in headers.items():
            output[key] = value
        output = json.dumps(output)
        async with aiofiles.open(self.headers_path, mode='w+') as f:
            await f.write(output)
            
    async def has_file_downloaded(self):
        # get headder file
        if os.path.isfile(self.headers_path):
            # if exists, load headers
            old_headers = await self.load_headers()
            old_headers = json.loads(old_headers)
            length = old_headers['Content-Length']
        else:
            return False
        # get file size
        file_size = os.path.getsize(self.download_path)
        if file_size == length:
            return True
        return False

    async def load_headers(self):
        async with aiofiles.open(self.headers_path, mode='r') as f:
            headers = await f.read()
            headers = json.loads(headers)
        return headers
    
    def partial_content(self, old_headers, have_size):
        print("old_headers",old_headers.keys())
        if 'Content-Length' in old_headers.keys() and "Accept-Ranges" in old_headers.keys():
            print("old_headers",old_headers.keys())
            length =old_headers['Content-Length']
            headers = {
                'Range': f'bytes={have_size}-{length}',
                'Accept-Ranges': 'bytes'
            }
            return headers
        return None

    def header_same_version(self, new_headers, old_headers):
        if new_headers == old_headers:
            return "same"
        if "ETag" in new_headers.keys() and "ETag" in old_headers.keys():
            print("new_headers",new_headers['ETag'], old_headers['ETag'])
            if new_headers['ETag'] == old_headers['ETag']:
                return "same"
            else:
                return "different"
        if "Last-Modified" in new_headers.keys() and "Last-Modified" in old_headers.keys():
            if new_headers['Last-Modified'] == old_headers['Last-Modified']:
                return "same"
            else:
                return "different"
        if "Content-Length" in new_headers.keys() and "Content-Length" in old_headers.keys():
            if new_headers['Content-Length'] != old_headers['Content-Length']:
                return "different"
        if "Content-MD5"  in new_headers.keys() and "Content-MD5"  in old_headers.keys():
            if new_headers['Content-MD5'] == old_headers['Content-MD5']:
                return "same"
            else:
                return "different"
        return "unknown"

async def main():
    ff = ResumableDownloader(
        'https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2', './latest-all.html')
    await ff.download()
asyncio.run(main())
