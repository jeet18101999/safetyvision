# SafetyVision 2.5

Desktop app (native window, no browser) that connects to an XVR/NVR, reads every channel and runs AI safety
analytics. Everything runs on this PC: YOLO models on the GPU and a local vision AI (Ollama, `qwen2.5vl`).

## Install

Run **`dist/SafetyVision-Setup.exe`**. It installs per user (no admin rights needed) and handles:
Python 3.11, PyTorch (CUDA when an NVIDIA GPU is present), all dependencies, the detection models,
**Ollama and the local vision model**, Start menu and desktop shortcuts, and an uninstaller in Windows Apps.
Running it again upgrades the app and keeps settings, events and trained models.

For development: `setup.ps1` creates `.venv`, `run.bat` starts the app, and
`.venv\Scripts\python.exe installer\build_installer.py` rebuilds the setup exe.

## Features

| Feature | Per camera | How it works |
|---|---|---|
| Man count | yes | Pose model + tracker, people on site, 24 h occupancy chart |
| Head pointer | display | Ring + pointer on each head (from face keypoints), toggle with boxes on the Live page |
| Man down | yes | Torso angle over N seconds, **confirmed by the local AI / learned model** before alerting |
| PPE | yes, per item | Helmet, vest, gloves, goggles, mask, safety shoes: learned per-item models, local AI when unsure |
| Fire / smoke | yes | Fire model + flicker heuristic, **every candidate verified by the local AI** |
| Static "person" check | global | Something shaped like a person that never moves is checked by the local AI; look-alikes (mannequins, coats, posters) are learned and ignored |
| Restricted zone | yes | Draw a polygon on the camera image; alert when someone's feet are inside |
| Crowd limit | yes | Alert when more than N people are on a camera |
| Camera offline | automatic | Alert when a channel has no video for 60 s |
| Alerts | - | Sound, Windows notification, tray icon, snapshot, CSV export, retention clean-up |

**Several recorders:** add any number of XVR / NVR / DVR units under *Settings > Recorders* (Dahua, Hikvision or a
custom RTSP URL). Each recorder has its own camera-name prefix (e.g. `DVR2-CH1`). The Live page can filter by recorder,
the Cameras page can enable or disable all cameras of a recorder at once, and the Dashboard shows health per recorder.

**Find devices automatically:** *Cameras > Discover devices...* (or *Settings > Recorders > Find devices on network*)
searches with ONVIF, Dahua and Hikvision discovery and scans camera ports on your subnets and any ranges you add
(needed for recorders behind a router). Enter the login once, click *Test login & detect channels* - the channel
count is read from the device (Dahua API, Hikvision ISAPI, ONVIF, or an RTSP probe) - then *Add ticked devices*.
IP cameras of any brand are added through ONVIF with their exact stream address.

**Names and places (sites, buildings, floors, rooms):** the left panel of the Cameras page holds your places as a
tree - *Add* a site, then buildings, floors and rooms inside it (buildings and floors are optional: a small site can
hold rooms directly). Right-click a place to rename, delete or give it a floor plan. Put cameras in a place with the
*Place* column or by selecting rows and pressing *Put selected cameras here*; clicking a place shows only its cameras.
**Click any camera row to see a live preview** of that camera on the right (a fresh picture is taken from the
recorder when monitoring is stopped), so you know which camera you are naming. On the **Site map** page pick any
place with a floor plan and click to place its cameras: markers show live status and people count and flash red on
alerts. Live and Events can be filtered by any place (a building shows all its floors and rooms), and the Dashboard
shows people per place. Older flat locations and room names are converted automatically.

Choose features per camera on the **Cameras** page. Only enabled features run, so switching off what a camera
doesn't need frees GPU and local-AI capacity for the others.

## Continuous learning (every model)

| Model | Learns from |
|---|---|
| Fire & smoke detector (YOLO) | Local-AI verdicts on every candidate, plus your reviews |
| Person verifier (classifier) | Local-AI checks of static people, walking people (automatic positives), your "Not a person" |
| Man-down verifier (classifier) | Local-AI checks of every man-down, plus your reviews |
| PPE: one classifier per item | Local-AI PPE checks, plus your reviews |

* **Review alerts** in *Events* (or the Live side panel): **Correct**, **False alarm** or **Not a person**
  (keys C / F / N). Your label counts 3× (configurable).
* **Corrections apply immediately**: false fire spots and "not a person" objects go into an ignore memory
  (*AI Learning → Ignored objects*), the matching samples are relabelled, and retraining is triggered sooner.
* A fire region the local AI rejects twice at the same spot is remembered permanently. A confident
  detection from the trained fire model is still always verified, so a real fire in that spot won't be missed.
* **Learning "not fire" without real fires.** Every candidate the local AI or you reject is kept as a "not fire"
  sample. Two things use them right away:
  * **Look-alike filter:** a new candidate whose colours match a rejected one from the same camera is skipped
    without asking the local AI. This works even when the object moves, such as a red jacket, a lamp or sun glare.
    On site data it caught 16 of 28 repeats and matched none of 1,800 fire images. Every 10th skip is still sent
    to the local AI as a safety check, and a confident detection from the fire model is never skipped.
  * **Fire model retraining:** this needs no real fire. *AI Learning → Fire & smoke detector → Train now* trains on
    your "not fire" images together with 540 reference fire images. These are simulated fires on this site's
    footage, shipped in `assets/fire_anchor`. The new model is deployed only if it raises **fewer false alarms on
    your "not fire" images** and still finds the reference fires on cameras it never trained on (mAP50 may drop
    by at most 0.05).
* Training runs in the background, one model at a time, on CPU when the GPU is busy. A new version is
  **deployed only if it scores at least as well as the current one** on the same validation images, and every
  version is kept (`models/<task>/versions`).
* Once a learned model is confident (≥ 0.85 by default) it answers alone and the local AI is only asked about
  uncertain cases, so the system gets faster as it learns.

## People, rooms and audio (2.3)

* **Recognise people by body (no face recognition).** Every tracked person gets appearance signatures from a
  body re-identification model (OpenCV Zoo "Youtu ReID", Apache-2.0, run on the GPU, about 1 ms per person). New people
  appear as *Person 17* on the **People** page (the sidebar shows how many are waiting, and a notification appears).
  Type a name once and the name tag follows that person on every camera (blue = named).
  Uncertain matches are checked by the local AI ("same person?"), and those answers and your names become training
  data. **Train recognition** learns a site-specific layer on top of the body model; it is only deployed if it
  identifies your named people better on held-out images. *Same person as...* merges two entries.
  Limits: identification uses clothing and build, so identical uniforms at a distance and a change of clothes are
  harder; the tracker can swap two people who cross, which is detected and split automatically.
* **Movement map.** Every sighting (person, camera, room, location, time in and out, walking path) is stored for
  30 days (Settings > Detection). The People page shows each person's timeline, their route across cameras drawn on
  the site map floor plan, and the path they walked inside each camera view.
* **Cameras in the same room.** Put cameras that look at the same room into the same *room* place on the Cameras
  page. A recognised person seen by several cameras of one room is counted **once**; the Dashboard shows unique
  people next to raw detections. (The automatic same-room finder of 2.3 was removed - it was not reliable enough.)
* **Audio.** Click the speaker on a camera tile to listen, one camera at a time, with volume and a level meter in the
  Live toolbar. Dahua XVRs send G.711 audio for every channel even without a microphone; the app says
  *no sound - camera may have no microphone?* when the channel is silent.
* **Privacy.** Recognising and tracking employees is personal data. Tell staff that the site uses CCTV analytics
  (a notice at the entrance or in the employee handbook), keep the retention short, and use *Delete* on the People page
  when someone asks. Recognition can be switched off in Settings > Detection.

## On / off switches and uninstalling

* **People: ON/OFF** (Live toolbar, also Settings > Detection) switches person detection off and on: people counting,
  man down, PPE, restricted zones and name tags. Fire detection keeps running. The choice is remembered.
* **AI analysis: ON/OFF** (Live toolbar) pauses all detection for view-only streaming (not remembered - it is back on
  after a restart, so alerts are never off by accident).
* **Uninstall:** `Uninstall SafetyVision.exe` in the install folder, the Start-menu shortcut, Windows *Apps &
  features*, or Settings > Alerts & app > *Uninstall SafetyVision...*. It removes only the files the installer listed
  in `installer\install.json` (never anything else in the folder), the shortcuts and the start-up entry. Tick
  *Also delete my data* to remove settings, events, snapshots, people and training data as well.
* The installer never spreads files over a drive root or a folder that already holds other things: choosing `D:\`
  installs into `D:\SafetyVision`.

## Live grid

* Any layout from 1 to 6 cameras per row; tiles are sized to fit the window (no sideways scrolling).
* **Rearrange cameras:** drag a tile onto another tile to swap them, or right-click a tile > *Move to position...*.
  The order is saved; right-click > *Reset camera order* restores it.
* **Sub stream for the grid** (Settings > Alerts & app): small tiles are shown from the recorder's light sub stream and
  the full main stream is used when a camera is opened; the AI always analyses the main stream. A sub stream is only
  used when it is at least 75 % as fast as the main stream, it never takes connections the main streams need, and the
  app tells you when the recorder's sub stream is too slow. Many XVRs ship with a slow sub stream (e.g. 352x288 at
  7 fps): set *Main Menu > Camera > Encode > Sub Stream* to D1 / 704x576, 15-25 fps for a smooth grid.

## Tools

* **Load test** (Dashboard > *Load test...*): records a clip from every camera and replays them as 16 / 24 / 32 / 48
  virtual cameras through the real pipeline. Reports the analysis rate per camera, GPU, CPU, RAM and local-AI delay,
  and the camera count this PC can handle. Also available as `python -m safetyvision.loadtest`.
* **Fire simulation** (AI Learning > *Fire simulation...*): renders realistic simulated fire and smoke on your own
  camera footage, checks it with the local AI, trains the fire detector and tests it on cameras it never saw and on
  real footage without fire, then runs one learning-loop retrain. *Use this model in the app* deploys it.
  Simulated fire is not real fire, so treat it as a head start.

## Notes
* Tested on Dahua DH-XVR4B16-I (16 channels, main stream 960×1080 "Lite", corrected to 16:9 automatically).
* GPU memory (8 GB): qwen2.5vl:7b uses ~5.5 GB. If analysis slows down, lower *GPU batch* in Settings or use
  `qwen2.5vl:3b`. The installer picks 3b automatically on GPUs with less than 8 GB.
* Only one copy of the app can run at a time (a second launch just brings the window to the front).
* The XVR password is stored in `config.yaml`, so keep the install folder private.

## Before you run it

- Copy `config.example.yaml` to `config.yaml` and enter your own NVR/XVR address, username and password.
- Trained models, site recordings and event data are not included in this repository. Standard YOLO weights download automatically on first run; the fire, PPE and man-down models must be trained on your own footage with the built-in trainer.

## Licence

GNU AGPL v3 (the app uses Ultralytics YOLO, which is AGPL-licensed). See [LICENSE](LICENSE).
