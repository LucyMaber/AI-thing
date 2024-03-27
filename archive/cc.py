import asyncio
import datetime
import io
import aioboto3
from warcio.archiveiterator import ArchiveIterator
from datetime import datetime
from archive.doamin_decoder_cdx import helper_cache_url
import os
import sys


class CC_Viewer:
    def __init__(self, url, mine, body) -> None:
        self.url = url
        self.mine = mine
        self.body = body

    def get_url(self):
        return self.url

    async def get_body(self):
        return self.body

    def get_mine(self):
        return self.mine


async def process_record(record, arc_url=None):
    try:
        warc_block_digest = record.rec_headers.get_header("WARC-Block-Digest")
        record_type = record.rec_headers.get_header("WARC-Type")
        record_date = record.rec_headers.get_header("WARC-Date")
        record_url = record.rec_headers.get_header("WARC-Target-URI")
        content_type = record.rec_headers.get_header("Content-Type")
        location = [warc_block_digest, record_url, arc_url]
        if record_url is None:
            return
        dt_object = datetime.strptime(record_date, "%Y-%m-%dT%H:%M:%SZ")
        data = CC_Viewer(record_url, content_type,
                         record.content_stream().read())
        await helper_cache_url(record_url, dt_object, data, "cc",
                               location=location, mime=content_type)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("exc_type",exc_type, fname, exc_tb.tb_lineno)
        print("process_record error:", e)


async def process_s3_object(session, bucket, key):
    async with session.client("s3") as s3:
        s3_ob = await s3.get_object(Bucket=bucket, Key=key)
        stream = s3_ob["Body"]
        data = await stream.read()
        ByteIO = io.BytesIO(data)

        try:
            for record in ArchiveIterator(ByteIO, arc2warc=True):
                await process_record(record, key)
        except Exception as e:
            print("process_s3_object", e)
        finally:
            ByteIO.close()


async def cc_scan():
    session = aioboto3.Session()
    bucket = "commoncrawl"
    prefixes = [
        "crawl-data/"
    ]
    async with session.resource("s3") as client:
        for prefix in prefixes:
            bucket_ = await client.Bucket(bucket)
            async for obj in bucket_.objects.filter(Prefix=prefix):
                if not obj.key.endswith(".warc.gz") and not obj.key.endswith(".arc.gz"):
                    continue

                await process_s3_object(session, bucket, obj.key)


# def main():
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(cc_scan())


# if __name__ == "__main__":
#     main()
