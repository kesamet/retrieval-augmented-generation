import os
import tempfile


def perform(func, filebytes, **kwargs):
    """Wrapper function to perform func for bytes file."""
    fh, temp_filename = tempfile.mkstemp()
    try:
        with open(temp_filename, "wb") as f:
            f.write(filebytes)
            f.flush()
            return func(f.name, **kwargs)
    finally:
        os.close(fh)
        os.remove(temp_filename)
