import aiofiles
import bz2


async def decompress_bz2_file(input_path):
    # Open the input BZ2-compressed file for asynchronous reading
    async with aiofiles.open(input_path, mode='rb') as input_file:
        decompressor = bz2.BZ2Decompressor()
        # Read and decompress data in chunks
        chunk = await input_file.read()
        if not chunk:
            return  # No need for the colon here
        return decompressor.decompress(chunk)
