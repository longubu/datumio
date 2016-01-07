"""
datumio.buffering
---------
Functions that wrap around generators to enable buffering. Buffering loads
the `n` generator calls into memory on one call.

Copied and modified from Kaggle National Data Science Bowl's 1st place winners.
Original in: https://github.com/benanne/kaggle-ndsb/blob/master/buffering.py

COPYRIGHT
---------
All contributions by Sander Dielman:
Copyright (c) 2015 Sander Dieleman
All rights reserved.

All contributions by Long Van Ho:
Copyright (c) 2015 Long Van Ho
All rights reserved.

All other contributions:
Copyright (c) 2015, the respective contributors.
All rights reserved.

LICENSE
---------
The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import multiprocessing as mp
import Queue
import threading


def buffered_gen_mp(source_gen, buffer_size=2):
    """Generator that runs a slow source generator in a separate process.

    Parameters
    -------
    buffer_size: the maximal number of items to pre-generate

    Yield
    -------
    data: buffered return from `source_gen`.

    Raises
    -------
    RunTimeERror: Minimal buffer sizeis less than 2. No need to use buffer.
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")

    buffer = mp.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate 1 extra element & block until there is room in the buffer.

    def _buffered_generation_process(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None)  # sentinel: signal the end of the iterator
        buffer.close()    # unfortunately this does not suffice as a signal:
        # if buffer.get() was called and subsequently the buffer is closed,
        # it will block forever.

    process = mp.Process(target=_buffered_generation_process,
                         args=(source_gen, buffer))
    process.start()

    for data in iter(buffer.get, None):
        yield data


def buffered_gen_threaded(source_gen, buffer_size=2):
    """Generator that runs a slow source generator in a separate thread.
    Beware of the GIL!

    Parameters
    -------
    buffer_size: the maximal number of items to pre-generate

    Yield
    -------
    data: buffered return from `source_gen`.
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")

    buffer = Queue.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate 1 extra element & block until there is room in the buffer.

    def _buffered_generation_thread(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None)  # sentinel: signal the end of the iterator

    thread = threading.Thread(target=_buffered_generation_thread,
                              args=(source_gen, buffer))
    thread.daemon = True
    thread.start()

    for data in iter(buffer.get, None):
        yield data
