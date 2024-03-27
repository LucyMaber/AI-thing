import os
import builtins
import struct
import zlib
import io
import gzip
from aiofile import async_open
from aiofile import AIOFile
BUFFER_SIZE = io.DEFAULT_BUFFER_SIZE  # Compressed data read chunk size


FTEXT, FHCRC, FEXTRA, FNAME, FCOMMENT = 1, 2, 4, 8, 16

READ, WRITE = 1, 2

_COMPRESS_LEVEL_FAST = 1
_COMPRESS_LEVEL_TRADEOFF = 6
_COMPRESS_LEVEL_BEST = 9


class BaseStream(io.BufferedIOBase):
    """Mode-checking helper functions."""

    async def _check_not_closed(self):
        if self.closed:
            raise ValueError("I/O operation on closed file")

    async def _check_can_read(self):
        if not self.readable():
            raise io.UnsupportedOperation("File not open for reading")

    async def _check_can_write(self):
        if not self.writable():
            raise io.UnsupportedOperation("File not open for writing")

    async def _check_can_seek(self):
        if not self.readable():
            raise io.UnsupportedOperation("Seeking is only supported "
                                          "on files open for reading")
        if not self.seekable():
            raise io.UnsupportedOperation("The underlying file object "
                                          "does not support seeking")


class DecompressReader(io.RawIOBase):
    """Adapts the decompressor API to a RawIOBase reader API"""

    def readable(self):
        return True

    def __init__(self, fp, decomp_factory, trailing_error=(), **decomp_args):
        self._fp = fp
        self._eof = False
        self._pos = 0  # Current offset in decompressed stream

        # Set to size of decompressed stream once it is known, for SEEK_END
        self._size = -1

        # Save the decompressor factory and arguments.
        # If the file contains multiple compressed streams, each
        # stream will need a separate decompressor object. A new decompressor
        # object is also needed when implementing a backwards seek().
        self._decomp_factory = decomp_factory
        self._decomp_args = decomp_args
        self._decompressor = self._decomp_factory(**self._decomp_args)

        # Exception class to catch from decompressor signifying invalid
        # trailing data to ignore
        self._trailing_error = trailing_error

    def close(self):
        self._decompressor = None
        return super().close()

    def seekable(self):
        return self._fp.seekable()

    def readinto(self, b):
        with memoryview(b) as view, view.cast("B") as byte_view:
            data = self.read(len(byte_view))
            byte_view[:len(data)] = data
        return len(data)

    def read(self, size=-1):
        if size < 0:
            return self.readall()

        if not size or self._eof:
            return b""
        data = None  # Default if EOF is encountered
        # Depending on the input data, our call to the decompressor may not
        # return any data. In this case, try again after reading another block.
        while True:
            if self._decompressor.eof:
                rawblock = (self._decompressor.unused_data or
                            self._fp.read(BUFFER_SIZE))
                if not rawblock:
                    break
                # Continue to next stream.
                self._decompressor = self._decomp_factory(
                    **self._decomp_args)
                try:
                    data = self._decompressor.decompress(rawblock, size)
                except self._trailing_error:
                    # Trailing data isn't a valid compressed stream; ignore it.
                    break
            else:
                if self._decompressor.needs_input:
                    rawblock = self._fp.read(BUFFER_SIZE)
                    if not rawblock:
                        raise EOFError("Compressed file ended before the "
                                       "end-of-stream marker was reached")
                else:
                    rawblock = b""
                data = self._decompressor.decompress(rawblock, size)
            if data:
                break
        if not data:
            self._eof = True
            self._size = self._pos
            return b""
        self._pos += len(data)
        return data

    def readall(self):
        chunks = []
        # sys.maxsize means the max length of output buffer is unlimited,
        # so that the whole input buffer can be decompressed within one
        # .decompress() call.
        while data := self.read(sys.maxsize):
            chunks.append(data)

        return b"".join(chunks)

    # Rewind the file to the beginning of the data stream.
    def _rewind(self):
        self._fp.seek(0)
        self._eof = False
        self._pos = 0
        self._decompressor = self._decomp_factory(**self._decomp_args)

    def seek(self, offset, whence=io.SEEK_SET):
        # Recalculate offset as an absolute file position.
        if whence == io.SEEK_SET:
            pass
        elif whence == io.SEEK_CUR:
            offset = self._pos + offset
        elif whence == io.SEEK_END:
            # Seeking relative to EOF - we need to know the file's size.
            if self._size < 0:
                while self.read(io.DEFAULT_BUFFER_SIZE):
                    pass
            offset = self._size + offset
        else:
            raise ValueError("Invalid value for whence: {}".format(whence))

        # Make it so that offset is the number of bytes to skip forward.
        if offset < self._pos:
            self._rewind()
        else:
            offset -= self._pos

        # Read and discard data until we reach the desired position.
        while offset > 0:
            data = self.read(min(io.DEFAULT_BUFFER_SIZE, offset))
            if not data:
                break
            offset -= len(data)

        return self._pos

    def tell(self):
        """Return the current file position."""
        return self._pos


async def _read_exact(fp, n):
    '''Read exactly *n* bytes from `fp`

    This method is required because fp may be unbuffered,
    i.e. return short reads.
    '''
    data = await fp.read(n)
    while len(data) < n:
        b = await fp.read(n - len(data))
        if not b:
            raise EOFError("Compressed file ended before the "
                           "end-of-stream marker was reached")
        data += b
    return data


async def _read_gzip_header(fp):
    '''Read a gzip header from `fp` and progress to the end of the header.

    Returns last mtime if header was present or None otherwise.
    '''
    magic = await fp.read(2)
    if magic == b'':
        return None

    if magic != b'\037\213':
        raise BadGzipFile('Not a gzipped file (%r)' % magic)

    (method, flag, last_mtime) = struct.unpack("<BBIxx", await _read_exact(fp, 8))
    if method != 8:
        raise BadGzipFile('Unknown compression method')

    if flag & FEXTRA:
        # Read & discard the extra field, if present
        extra_len, = struct.unpack("<H", await _read_exact(fp, 2))
        await _read_exact(fp, extra_len)
    if flag & FNAME:
        # Read and discard a null-terminated string containing the filename
        while True:
            s = fp.read(1)
            if not s or s == b'\000':
                break
    if flag & FCOMMENT:
        # Read and discard a null-terminated string containing a comment
        while True:
            s = fp.read(1)
            if not s or s == b'\000':
                break
    if flag & FHCRC:
        await _read_exact(fp, 2)     # Read & discard the 16-bit header CRC
    return last_mtime


class _PaddedFile:
    """Minimal read-only file object that prepends a string to the contents
    of an actual file. Shouldn't be used outside of gzip.py, as it lacks
    essential functionality."""

    def __init__(self, f, prepend=b''):
        self._buffer = prepend
        self._length = len(prepend)
        self.file = f
        self._read = 0

    async def read(self, size):
        if self._read is None:
            return await self.file.read(size)
        if self._read + size <= self._length:
            read = self._read
            self._read += size
            return self._buffer[read:self._read]
        else:
            read = self._read
            self._read = None
            a = await self.file.read(size-self._length+read)
            return self._buffer[read:] + a

    async def prepend(self, prepend=b''):
        if self._read is None:
            self._buffer = prepend
        else:  # Assume data was read since the last prepend() call
            self._read -= len(prepend)
            return
        self._length = len(self._buffer)
        self._read = 0

    async def seek(self, off):
        self._read = None
        self._buffer = None
        return await self.file.seek(off)

    async def seekable(self):
        return True  # Allows fast-forwarding even in unseekable streams


class BadGzipFile(OSError):
    """Exception raised in some cases for invalid gzip files."""


class _GzipReader(DecompressReader):
    def __init__(self, fp):
        super().__init__(_PaddedFile(fp), zlib.decompressobj,
                         wbits=-zlib.MAX_WBITS)
        # Set flag indicating start of a new member
        self._new_member = True
        self._last_mtime = None

    def _init_read(self):
        self._crc = zlib.crc32(b"")
        self._stream_size = 0  # Decompressed size of unconcatenated stream

    async def _read_gzip_header(self):
        last_mtime = await _read_gzip_header(self._fp)
        if last_mtime is None:
            return False
        self._last_mtime = last_mtime
        return True

    async def read(self, size=-1):
        if size < 0:
            return self.readall()
        # size=0 is special because decompress(max_length=0) is not supported
        if not size:
            return b""

        # For certain input data, a single
        # call to decompress() may not return
        # any data. In this case, retry until we get some data or reach EOF.
        while True:
            if self._decompressor.eof:
                # Ending case: we've come to the end of a member in the file,
                # so finish up this member, and read a new gzip header.
                # Check the CRC and file size, and set the flag so we read
                # a new member
                await self._read_eof()
                self._new_member = True
                self._decompressor = self._decomp_factory(
                    **self._decomp_args)

            if self._new_member:
                # If the _new_member flag is set, we have to
                # jump to the next member, if there is one.
                self._init_read()
                if not await self._read_gzip_header():
                    self._size = self._pos
                    return b""
                self._new_member = False

            # Read a chunk of data from the file
            buf = await self._fp.read(io.DEFAULT_BUFFER_SIZE)
            uncompress = self._decompressor.decompress(buf, size)
            if self._decompressor.unconsumed_tail != b"":
                await self._fp.prepend(self._decompressor.unconsumed_tail)
            elif self._decompressor.unused_data != b"":
                # Prepend the already read bytes to the fileobj so they can
                # be seen by _read_eof() and _read_gzip_header()
                await self._fp.prepend(self._decompressor.unused_data)

            if uncompress != b"":
                break
            if buf == b"":
                raise EOFError("Compressed file ended before the "
                               "end-of-stream marker was reached")

        await self._add_read_data(uncompress)
        self._pos += len(uncompress)
        return uncompress

    async def _add_read_data(self, data):
        self._crc = zlib.crc32(data, self._crc)
        self._stream_size = self._stream_size + len(data)

    async def _read_eof(self):
        # We've read to the end of the file
        # We check that the computed CRC and size of the
        # uncompressed data matches the stored values.  Note that the size
        # stored is the true file size mod 2**32.
        crc32, isize = struct.unpack("<II", await _read_exact(self._fp, 8))
        if crc32 != self._crc:
            raise BadGzipFile("CRC check failed %s != %s" % (hex(crc32),
                                                             hex(self._crc)))
        elif isize != (self._stream_size & 0xffffffff):
            raise BadGzipFile("Incorrect length of data produced")

        # Gzip files can be padded with zeroes and still have archives.
        # Consume all zero bytes and set the file position to the first
        # non-zero byte. See http://www.gzip.org/#faq8
        c = b"\x00"
        while c == b"\x00":
            c = await self._fp.read(1)
        if c:
            await self._fp.prepend(c)

    async def _rewind(self):
        super()._rewind()
        self._new_member = True


class AioGzipFile:
    """The GzipFile class simulates most of the methods of a file object with
    the exception of the truncate() method.

    This class only supports opening files in binary mode. If you need to open a
    compressed file in text mode, use the gzip.open() function.

    """

    # Overridden with internal file object to be closed, if only a filename
    # is passed in
    myfileobj = None

    def __init__(self, fileobj, mode="r", compresslevel=gzip._COMPRESS_LEVEL_BEST, mtime=None):
        self.mode_ = mode
        self.compresslevel = compresslevel
        self.fileobj = fileobj
        self.mtime_ = mtime
        self.filename_ = None

        if self.mode_ and ('t' in self.mode_ or 'U' in self.mode_):
            raise ValueError("Invalid mode: {!r}".format(self.mode_))
        if self.mode_ and 'b' not in self.mode_:
            self.mode_ += 'b'

        elif self.mode_.startswith(('w', 'a', 'x')):
            if origmode is None:
                import warnings
                warnings.warn(
                    "GzipFile was opened for writing, but this will "
                    "change in future Python releases.  "
                    "Specify the mode argument for opening it for writing.",
                    FutureWarning, 2)
            self.mode = WRITE
            self._init_write(self.filename)
            self.compress = zlib.compressobj(compresslevel,
                                             zlib.DEFLATED,
                                             -zlib.MAX_WBITS,
                                             zlib.DEF_MEM_LEVEL,
                                             0)
            self._write_mtime = mtime
        else:
            raise ValueError("Invalid mode: {!r}".format(mode))

    async def __aenter__(self):
        # if self.fileobj is None:
        #     self.fileobj = await (AIOFile(self.filename_, 'rb')).__aenter__()
        if self.filename_ is None:
            self.filename_ = getattr(self.fileobj, 'name', '')
            if not isinstance(self.filename_, (str, bytes)):
                self.filename_ = ''
        else:
            self.filename_ = os.fspath(self.filename_)
        # origmode = self.mode_
        if self.mode_ is None:
            self.mode_ = getattr(fileobj, 'mode', 'rb')
        if self.mode_ == WRITE:
            await self._write_gzip_header(self.compresslevel)

        if self.mode_.startswith('r'):
            self.mode = READ
            self._buffer = _GzipReader(self.fileobj)
            # self._buffer = io.BufferedReader(raw)
            self.name = self.filename_
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @ property
    def filename(self):
        import warnings
        warnings.warn("use the name attribute", DeprecationWarning, 2)
        if self.mode_ == WRITE and self.name[-3:] != ".gz":
            return self.name + ".gz"
        return self.name

    @ property
    def mtime(self):
        """Last modification time read from stream, or None"""
        return self._buffer.raw._last_mtime

    def __repr__(self):
        s = repr(self.fileobj)
        return '<gzip ' + s[1:-1] + ' ' + hex(id(self)) + '>'

    def _init_write(self, filename):
        self.name = filename
        self.crc = zlib.crc32(b"")
        self.size = 0
        self.writebuf = []
        self.bufsize = 0
        self.offset = 0  # Current file offset for seek(), tell(), etc

    async def _write_gzip_header(self, compresslevel):
        self.fileobj.write(b'\037\213')             # magic header
        self.fileobj.write(b'\010')                 # compression method
        try:
            # RFC 1952 requires the FNAME field to be Latin-1. Do not
            # include filenames that cannot be represented that way.
            fname = os.path.basename(self.name)
            if not isinstance(fname, bytes):
                fname = fname.encode('latin-1')
            if fname.endswith(b'.gz'):
                fname = fname[:-3]
        except UnicodeEncodeError:
            fname = b''
        flags = 0
        if fname:
            flags = FNAME
        self.fileobj.write(chr(flags).encode('latin-1'))
        mtime = self._write_mtime
        if mtime is None:
            mtime = time.time()
        write32u(self.fileobj, int(mtime))
        if compresslevel == _COMPRESS_LEVEL_BEST:
            xfl = b'\002'
        elif compresslevel == _COMPRESS_LEVEL_FAST:
            xfl = b'\004'
        else:
            xfl = b'\000'
        self.fileobj.write(xfl)
        self.fileobj.write(b'\377')
        if fname:
            self.fileobj.write(fname + b'\000')

    async def write(self, data):
        # await self._check_not_closed()
        if self.mode != WRITE:
            import errno
            raise OSError(errno.EBADF, "write() on read-only GzipFile object")

        if self.fileobj is None:
            raise ValueError("write() on closed GzipFile object")

        if isinstance(data, (bytes, bytearray)):
            length = len(data)
        else:
            # accept any data that supports the buffer protocol
            data = memoryview(data)
            length = data.nbytes

        if length > 0:
            self.fileobj.write(self.compress.compress(data))
            self.size += length
            self.crc = zlib.crc32(data, self.crc)
            self.offset += length

        return length

    async def read(self, size=-1):
        # self._check_not_closed()
        if self.mode != READ:
            import errno
            raise OSError(errno.EBADF, "read() on write-only GzipFile object")
        return await self._buffer.read(size)

    async def read1(self, size=-1):
        """Implements BufferedIOBase.read1()

        Reads up to a buffer's worth of data if size is negative."""
        # self._check_not_closed()
        if self.mode != READ:
            import errno
            raise OSError(errno.EBADF, "read1() on write-only GzipFile object")

        if size < 0:
            size = io.DEFAULT_BUFFER_SIZE
        return await self._buffer.read1(size)

    async def peek(self, n):
        # self._check_not_closed()
        if self.mode != READ:
            import errno
            raise OSError(errno.EBADF, "peek() on write-only GzipFile object")
        return await self._buffer.peek(n)

    @ property
    def closed(self):
        return self.fileobj is None

    def close(self):
        fileobj = self.fileobj
        if fileobj is None:
            return
        self.fileobj = None
        try:
            if self.mode == WRITE:
                fileobj.write(self.compress.flush())
                write32u(fileobj, self.crc)
                # self.size may exceed 2 GiB, or even 4 GiB
                write32u(fileobj, self.size & 0xffffffff)
            elif self.mode == READ:
                self._buffer.close()
        finally:
            myfileobj = self.myfileobj
            if myfileobj:
                self.myfileobj = None
                myfileobj.close()

    async def flush(self, zlib_mode=zlib.Z_SYNC_FLUSH):
        # self._check_not_closed()
        if self.mode == WRITE:
            # Ensure the compressor's buffer is flushed
            self.fileobj.write(self.compress.flush(zlib_mode))
            self.fileobj.flush()

    def fileno(self):
        """Invoke the underlying file object's fileno() method.

        This will raise AttributeError if the underlying file object
        doesn't support fileno().
        """
        return self.fileobj.fileno()

    async def rewind(self):
        '''Return the uncompressed stream file position indicator to the
        beginning of the file'''
        if self.mode != READ:
            raise OSError("Can't rewind in write mode")
        await self._buffer.seek(0)

    def readable(self):
        return self.mode == READ

    def writable(self):
        return self.mode == WRITE

    def seekable(self):
        return True

    async def seek(self, offset, whence=io.SEEK_SET):
        if self.mode == WRITE:
            if whence != io.SEEK_SET:
                if whence == io.SEEK_CUR:
                    offset = self.offset + offset
                else:
                    raise ValueError('Seek from end not supported')
            if offset < self.offset:
                raise OSError('Negative seek in write mode')
            count = offset - self.offset
            chunk = b'\0' * 1024
            for i in range(count // 1024):
                self.write(chunk)
            self.write(b'\0' * (count % 1024))
        elif self.mode == READ:
            # self._check_not_closed()
            return self._buffer.seek(offset, whence)

        return self.offset

    async def readline(self, size=-1):
        # self._check_not_closed()
        return await self._buffer.readline(size)
