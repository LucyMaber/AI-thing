from utility.task import task_init
import asyncio
from utility.db import make_veter_db, make_mogo_db
from pull.wikidata import async_load_wikidata_file


async def main():
    await make_veter_db()
    await make_mogo_db()
    await async_load_wikidata_file()

asyncio.run(main())
