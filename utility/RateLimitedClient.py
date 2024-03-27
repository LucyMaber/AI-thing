import asyncio
import time
import aiohttp
import urllib.parse
import urllib.robotparser
from collections import OrderedDict
from asyncio import Lock


class HttpRequestManager:
    def __init__(self, max_domains=60):
        self.max_domains = max_domains
        self.domain_info = OrderedDict()

    async def fetch_text(self, url, headers=None):
        domain = self.extract_domain(url)
        await self.add_domain_if_needed(domain)
        async with self.domain_info[domain]["lock"]:
            for i in range(10):
                await self.wait_for_domain(url)
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url) as response:
                        if response.status == 429:
                            await self.handle_rate_limit(domain, response, i)
                            continue
                        return await response.text(), response.status
            return None, None

    async def fetch_binary(self, url, headers=None):
        domain = self.extract_domain(url)
        await self.add_domain_if_needed(domain)
        async with self.domain_info[domain]["lock"]:
            for i in range(10):
                await self.wait_for_domain(url)
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url) as response:
                        if response.status == 429 and response.status == 403:
                            await self.handle_rate_limit(
                                domain, response, i
                            ), response.status
                            continue
                        return await response.read(), response.status
            return None, None

    async def wait_for_domain(self, url):
        domain = self.extract_domain(url)
        
        await asyncio.sleep(self.get_delay(domain))

        return 

    async def add_domain_if_needed(self, domain):
        if domain not in self.domain_info:
            await self.add_domain(domain)

    async def handle_rate_limit(self, domain, response, i):
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            await asyncio.sleep(int(retry_after))
        else:
            await asyncio.sleep(5 * i)  # Default wait time

    async def add_domain(self, domain):
        crawl_delay, request_rate, parser = await self.parse_robots_txt(domain)
        self.domain_info[domain] = {
            "last_request_time": 0,
            "crawl_delay": crawl_delay,
            "request_rate": request_rate,
            "robots": parser,
            "lock": Lock(),
        }

    def can_make_request(self, url):
        domain = self.extract_domain(url)
        path = urllib.parse.urlparse(url).path

        if domain not in self.domain_info:
            return True

        current_time = asyncio.get_event_loop().time()
        last_request_time = self.domain_info[domain]["last_request_time"]
        crawl_delay = self.domain_info[domain]["crawl_delay"]
        request_rate = self.domain_info[domain]["request_rate"]
        robots_parser = self.domain_info[domain]["robots"]

        if current_time - last_request_time < crawl_delay or (
            request_rate and current_time - last_request_time < 1 / request_rate
        ):
            return False

        if robots_parser and not robots_parser.can_fetch("*", path):
            return False

        self.domain_info[domain]["last_request_time"] = current_time
        return True

    def get_delay(self, domain):
        current_time = time.time()
        last_request_time = self.domain_info[domain]["last_request_time"]
        crawl_delay = self.domain_info[domain]["crawl_delay"]
        request_rate = self.domain_info[domain]["request_rate"]

        # Calculate the remaining time until the next allowed request
        remaining_crawl_delay = last_request_time + crawl_delay - current_time
        remaining_request_rate = (
            (1 / request_rate) - (current_time - last_request_time)
            if request_rate
            else 0
        )

        # Return the maximum of the two remaining times
        return max(remaining_crawl_delay, remaining_request_rate, 0)

    async def parse_robots_txt(self, domain):
        robots_txt_url = f"http://{domain}/robots.txt"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(robots_txt_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        parser = urllib.robotparser.RobotFileParser()
                        parser.parse(content)
                        crawl_delay = parser.crawl_delay("*")
                        request_rate = parser.request_rate("*")
                        return (
                            crawl_delay if crawl_delay is not None else 1,
                            request_rate,
                            parser,
                        )
        except aiohttp.ClientError:
            pass
        return 1, None, None
    def extract_domain(self, url):
        return urllib.parse.urlparse(url).netloc


H = HttpRequestManager()
 

def GET_HTTP_SAFE():
    return H

async def main():
    data =await H.fetch_text("https://www.google.com") 
    print(data)
    pass

if __name__ == "__main__":
    asyncio.run(main())