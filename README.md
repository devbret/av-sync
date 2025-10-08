# AV-Sync

This application builds a beat-synced music video from sound and video source files. To accomplish this AV-Sync first analyzes the input audio with Librosa to detect onsets, segments it and computes features to classify each segment as slow, medium or fast.

It also scores the video source files with OpenCV by measuring frame-to-frame motion. AV-Sync then picks diverse start offsets whose motion “bucket” matches each audio segment.

Finally, it generates an FFmpeg filter graph which trims, zooms/pans, scales, centers, concatenates the clips over the audio and renders the result. AV-Sync also writes a JSON manifest describing all measurements and placements.

## Please Note

AV-Sync is a resource heavy application, especially with regards to CPU and RAM. So please start small with your projects and have a good time.
