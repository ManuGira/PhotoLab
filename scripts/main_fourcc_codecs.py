import cv2
from pprint import pprint

"""
There is no way to actually enumerate available fourcc codecs in OpenCV.
This is quite unfortunate, since due to licensing issues OpenCV codec packaging differs between distributors.
Using isOpened() on VideoWriter tells you if a encoder could be initialized successfully.
With a given list of fourcc codecs you can do something like this though:
"""
def is_fourcc_available(codec):
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        temp_video = cv2.VideoWriter('temp.mkv', fourcc, 30, (640, 480), isColor=True)
        return temp_video.isOpened()
    except:
        return False


def enumerate_fourcc_codecs():
    codecs_to_test = ["DIVX", "XVID", "MJPG", "X264", "WMV1", "WMV2", "FMP4",
                      "mp4v", "avc1", "I420", "IYUV", "mpg1", "H264"]
    available_codecs = []
    for codec in codecs_to_test:
        available_codecs.append((codec, is_fourcc_available(codec)))
    return available_codecs


if __name__ == "__main__":
    codecs = enumerate_fourcc_codecs()
    print("Available FourCC codecs:")
    pprint(codecs)