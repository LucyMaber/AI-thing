import os
import io
import zstandard
import asyncio
from zstandard import ZstdError, lib, ffi

class AsyncZstdDecompressionReader:
    """Read only decompressor that pull uncompressed data from another stream.

    This type provides a read-only stream interface for performing transparent
    decompression from another stream or data source. It conforms to the
    ``io.RawIOBase`` interface. Only methods relevant to reading are
    implemented.

    Usage example:

    ```python
    async def example():
        with open(path, 'rb') as fh:
            dctx = zstandard.ZstdDecompressor()
            reader = AsyncZstdDecompressionReader(dctx.stream_reader(fh))
            while True:
                chunk = await reader.read(16384)
                if not chunk:
                    break
                # Do something with decompressed chunk.
    ```

    """

    def __init__(self, source):
        self._source = source
        self._bytes_decompressed = 0
        self._finished_output = False
        self._in_buffer = ffi.new("ZSTD_inBuffer *")
        self._source_buffer = None

    async def _read_input(self):
        if self._source_buffer is not None:
            return

        if hasattr(self._source, "read"):
            data = await self._source.read(self._read_size)
            if not data:
                return
            self._source_buffer = ffi.from_buffer(data)
            self._in_buffer.src = self._source_buffer
            self._in_buffer.size = len(self._source_buffer)
            self._in_buffer.pos = 0
        else:
            self._source_buffer = ffi.from_buffer(self._source)
            self._in_buffer.src = self._source_buffer
            self._in_buffer.size = len(self._source_buffer)
            self._in_buffer.pos = 0

    async def _decompress_into_buffer(self, out_buffer):
        zresult = lib.ZSTD_decompressStream(
            self._decompressor._dctx, out_buffer, self._in_buffer
        )

        if self._in_buffer.pos == self._in_buffer.size:
            self._in_buffer.src = ffi.NULL
            self._in_buffer.pos = 0
            self._in_buffer.size = 0
            self._source_buffer = None

            if not hasattr(self._source, "read"):
                self._finished_input = True

        if lib.ZSTD_isError(zresult):
            raise ZstdError("zstd decompress error: %s" % _zstd_error(zresult))

        return out_buffer.pos and (
            out_buffer.pos == out_buffer.size
            or zresult == 0
            and not self._read_across_frames
        )

    async def read(self, size=-1):
        if self._finished_output or size == 0:
            return b""

        if size < 0:
            return await self.readall()

        dst_buffer = ffi.new("char[]", size)
        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = dst_buffer
        out_buffer.size = size
        out_buffer.pos = 0

        await self._read_input()
        if await self._decompress_into_buffer(out_buffer):
            self._bytes_decompressed += out_buffer.pos
            return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

        while not self._finished_input:
            await self._read_input()
            if await self._decompress_into_buffer(out_buffer):
                self._bytes_decompressed += out_buffer.pos
                return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

        self._bytes_decompressed += out_buffer.pos
        return ffi.buffer(out_buffer.dst, out_buffer.pos)[:]

    async def readall(self):
        chunks = []
        while True:
            chunk = await self.read(1048576)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    async def readinto(self, b):
        if self._finished_output:
            return 0

        dest_buffer = ffi.from_buffer(b)
        ffi.memmove(b, b"", 0)
        out_buffer = ffi.new("ZSTD_outBuffer *")
        out_buffer.dst = dest_buffer
        out_buffer.size = len(dest_buffer)
        out_buffer.pos = 0

        await self._read_input()
        if await self._decompress_into_buffer(out_buffer):
            self._bytes_decompressed += out_buffer.pos
            return out_buffer.pos

        while not self._finished_input:
            await self._read_input()
            if await self._decompress_into_buffer(out_buffer):
                self._bytes_decompressed += out_buffer.pos
                return out_buffer.pos

        self._bytes_decompressed += out_buffer.pos
        return out_buffer.pos


import aiofile
import asyncio
import aiohttp

async def read_zstd_file(filename):
    async with aiofile.async_open(filename, 'rb') as fh:
        # Assuming you have a ZstdDecompressor instance created
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        async_zstd_reader = AsyncZstdDecompressionReader(reader)
        data = await async_zstd_reader.readall()
        return data

async def main():
    filename = '/home/lucy/Code/code_reddit/data/pullshift/submissions/RS_2022-08.zst'
    data = await read_zstd_file(filename)
    print(data)