# AV-Sync

Builds a beat-synced music video from an input audio track and a folder of source video clips.

## Overview

AV-Sync analyzes source audio files with Librosa, separates harmonic and percussive content, detects onsets and divides the track into musical segments using different segmentation modes.

For each audio segment, AV-Sync computes detailed rhythmic, loudness and novelty features, then classifies the segment as slow, medium or fast. It also analyzes each candidate video with `OpenCV` by sampling motion over time, building timelines and grouping clips into motion buckets so the visual pacing can follow the energy of the music.

Finally, AV-Sync selects diverse video regions to best match each audio segment’s motion profile and timing. It then builds an `FFmpeg` filter graph which trims clips, applies zooms, pans, pulse effects, motion trails and audio mapping before rendering the final synchronized video.

The program also writes a JSON manifest containing the audio analysis, clip choices, timing offsets, feature measurements and placement metadata.

## Basic Setup Instructions

Below are the required software programs and instructions for installing and using this application on a Linux machine.

### Programs Needed

- [Git](https://git-scm.com/downloads)

- [Python](https://www.python.org/downloads/)

### Steps For Use

1. Install the above programs

2. Open a terminal

3. Clone this repository: `git clone git@github.com:devbret/av-sync.git`

4. Navigate to the repo's directory: `cd av-sync`

5. Create a virtual environment: `python3 -m venv venv`

6. Activate your virtual environment: `source venv/bin/activate`

7. Install the needed dependencies: `pip install -r requirements.txt`

8. Place your source files into the `input` and `videos` directories

9. Run the program: `python3 app.py`

10. Locate your results in the `output` directory

11. Exit the virtual environment: `deactivate`

## Other Considerations

This project repo is intended to demonstrate an ability to do the following:

- Turn an audio track and a folder of video clips into a beat-synced music video

- Analyze tempo, onsets, energy and tonal features of the source audio file

- Score videos by motion and select clip regions to match the pace and intensity of each audio segment

- Save a JSON manifest of the audio analysis, clip choices, timing and placement data

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).

### Please Note

AV-Sync can be an exceptionally resource heavy application, particularly with regards to the CPU and RAM. So please start small with your projects and have a good time.
