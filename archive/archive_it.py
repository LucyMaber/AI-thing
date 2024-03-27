import aiohttp
import asyncio

import httpx


async def fetch_data(domain, p, page):
    url = f"https://archive-it.org/explore?q={domain}&show=ArchivedPages&page={page}&totalResultCount=826&p={p}"

    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "en-GB,en;q=0.9",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "sec-ch-ua": "\"Not.A/Brand\";v=\"8\", \"Chromium\";v=\"114\", \"Google Chrome\";v=\"114\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Linux\"",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1"
    }

    referrer = f"https://archive-it.org/explore?q={domain}"

    async with httpx.AsyncClient() as session:
        response = await session.get(url, headers=headers, referrer=referrer,
                               referrer_policy="strict-origin-when-cross-origin", raise_for_status=True)
        response_text = await response.text
    return response_text


async def main():
    p = "reddit.com/domains/"
    page = "1"
    data = await fetch_data(p, page)

if __name__ == "__main__":
    asyncio.run(main())
