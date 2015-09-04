"""Basic interface for distributed files

Currently, only supports files backed by an S3 bucket.
"""

import cStringIO
import math
import os

from boto import s3

MODE_READ = "r"
MODE_WRITE = "w"

S3_CHUNK_SIZE = 64*1024*1024  # 64 MB


def parse_s3_filename(filename):
    parts = filename.split("/")
    assert len(parts) >= 4
    assert parts[0] == ""
    return parts[1], parts[2], "/".join(parts[3:])


class S3File(object):
    """File stored on S3.

    Example: /s3/us-west-2/distopt/path/to/file
    """
    def __init__(self, filename, mode):
        region, bucket_name, path = parse_s3_filename(filename)
        self.conn = s3.connect_to_region(region)
        self.bucket = self.conn.get_bucket(bucket_name)
        self.write_offset = 0
        self.mode = mode

        if self.mode == MODE_WRITE:
            self.write_called = False
            self.upload = self.bucket.initiate_multipart_upload(path)
        elif self.mode == MODE_READ:
            self.key = self.bucket.get_key(path)

    def read(self):
        return self.key.get_contents_as_string()

    def write(self, data):
        assert self.mode == MODE_WRITE

        # TODO(mwytock): Fix this
        assert not self.write_called, "multiple calls to write() not supported"
        self.write_called = True

        # TODO(mwytock): Make this operate on file-like objects rather than just
        # buffers. This is a little tricky to do in a traditional file-like
        # interface though because S3 actually uploads in chunks
        size = len(data)
        chunk_count = int(math.ceil(size / float(S3_CHUNK_SIZE)))

        for i in range(chunk_count):
            ii = i*S3_CHUNK_SIZE
            jj = min(ii + S3_CHUNK_SIZE, size)
            fp = cStringIO.StringIO(data[ii:jj])
            self.upload.upload_part_from_file(fp, part_num=i+1)

    def close(self):
        if self.mode == MODE_WRITE:
            self.upload.complete_upload()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


# TODO(mwytock): Is this the best way to do this?
py_open = open

def open(filename, mode=MODE_READ):
    assert mode == MODE_READ or MODE_WRITE

    assert filename[0] == "/"
    i = filename.find("/", 1)
    file_type = filename[1:i]
    sub_path = filename[i:]

    if file_type == "s3":
        return S3File(sub_path, mode)
    if file_type == "local":
        if mode == MODE_WRITE:
            dirname = os.path.dirname(sub_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        return py_open(sub_path, mode)

    raise Exception("Unkonwn file type: %s", file_type)
