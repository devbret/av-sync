# AV-Sync

This application builds a beat-synced music video from sound and video source files. To accomplish this AV-Sync first analyzes the input audio with Librosa to detect onsets, segments it and computes features to classify each segment as slow, medium or fast.

It also scores the video source files with OpenCV by measuring frame-to-frame motion. AV-Sync then picks diverse start offsets whose motion “bucket” matches each audio segment.

Finally, it generates an FFmpeg filter graph which trims, zooms/pans, scales, centers, concatenates the clips over the audio and renders the result. AV-Sync also writes a JSON manifest describing all measurements and placements.

## Set Up Instructions

Below are the required software programs and instructions for installing and using this application.

### Programs Needed

- [Git](https://git-scm.com/downloads)

- [Python](https://www.python.org/downloads/)

### Steps For Use

1. Install the above programs

2. Open a terminal

3. Clone this repository using `git` by running the following command: `git clone git@github.com:devbret/av-sync.git`

4. Navigate to the repo's directory by running: `cd av-sync`

5. Create a virtual environment with this command: `python3 -m venv venv`

6. Activate your virtual environment using: `source venv/bin/activate`

7. Install the needed dependencies for running the script: `pip install -r requirements.txt`

8. Place your source files into the `input` and `videos` directories

9. Run the program using this command: `python3 app.py`

10. Locate your results in the `output` directory

11. To exit the virtual environment (venv), type this command in the terminal: `deactivate`

## Please Note

AV-Sync is an exceptionally resource heavy application, particularly with regards to the CPU and RAM. So please start small with your projects and have a good time.

## Other Considerations

This project repo is intended to demonstrate an ability to do the following:

- Analyze an audio track, split it into beat segments and measure detailed features

- Automate stitching video clips together into a music-synced visual edit

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).
