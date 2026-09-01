"""Pick a video codec from the output path's extension.

Every post-processing stage re-encodes the whole clip, and mp4v is lossy enough that the
generational damage is measurable in perception: two null re-encode passes of a Town04 night
clip -- changing not one pixel on purpose -- dropped Vision Pilot's lead-vehicle detection from
50.3% of frames to 41.7%. The six-stage night chain therefore threw away real signal before the
clip was ever scored.

FFV1 in an .avi container round-trips bit-exactly here (verified mean abs error 0.0), so
intermediate stages can be lossless and only the delivered file needs compressing.

Callers opt in simply by naming the intermediate .avi; anything else keeps mp4v, so existing
chains behave exactly as before.
"""
import cv2

LOSSLESS_EXT = ('.avi', '.mkv')


def fourcc_for(path):
    p = str(path).lower()
    return cv2.VideoWriter_fourcc(*('FFV1' if p.endswith(LOSSLESS_EXT) else 'mp4v'))
