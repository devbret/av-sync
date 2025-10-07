# AV-Sync

This program builds a beat-synced music video from a folder of source clips. It analyzes the input audio with librosa to detect onsets, segments it and computes features to classify each segment as slow, medium or fast.

It also scores source video files with OpenCV by measuring frame-to-frame motion. AV-Sync then picks diverse start offsets whose motion “bucket” matches each audio segment.

Finally, it generates an FFmpeg filter graph that trims, zooms/pans, scales, centers, concatenates the clips under the audio and renders the result. AV-Sync also writes a JSON manifest describing all placements.

## Please Note

AV-Sync is an resource heavy application, especially with regards to CPU and RAM. So please start small with your projects and have a good time.
