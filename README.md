# Project Chimera: AI Doodle Recognition

Draw on a canvas and a small convolutional neural network (CNN) guesses what you drew. A chart shows how
its confidence changes as you add more detail.

Built with **Python, PyTorch, Streamlit, NumPy, Pillow, Pandas, Plotly and Pytest**. It runs entirely
on your own machine: no API key, no paid service, no JavaScript front-end.

<!-- Add screenshots here: docs/draw.png, docs/lab.png, docs/model.png
![Draw tab](docs/draw.png) -->

## Features

| Tab | What it does |
|---|---|
| **Draw** | Drawing canvas, clear button, brush-size slider, main prediction, top-5 guesses with confidence, probability chart for all 10 classes, the processed 28 × 28 input, and a prediction timeline that records how the answer changes as the drawing develops |
| **Experiment Lab** | Feeds your drawing through variations (mirrored, rotated, inverted, shifted, half size, thicker/thinner strokes, noise, no crop) and shows whether the answer holds up. An *occlusion map* hides parts of the drawing to reveal which regions the model relies on |
| **Model** | Parameter count, test accuracy, top-3 accuracy, macro F1, architecture, training curves, confusion matrix and per-class scores |

## How it works

```
canvas (320x320 RGBA)
   |  ui/canvas.py            streamlit-drawable-canvas, white ink on black
   v
preprocessing (data/preprocessing.py)
   grayscale -> fix background polarity -> remove background level -> bounding box
   -> crop -> keep aspect ratio -> scale longest side to 20 px -> centre in 28x28 -> normalise
   v
CNN (models/cnn.py)          3 x [Conv3x3 - BatchNorm - ReLU - MaxPool] -> Linear(128) -> Dropout -> 10 logits
   v
softmax -> Prediction (models/prediction.py) -> timeline, charts, Experiment Lab (ui/)
```

**Data.** [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) by Google (licence
CC BY 4.0). The downloader fetches the public `numpy_bitmap` files, but only the first 5,000 drawings of
each of the 10 classes, using an HTTP Range request (about 4 MB per class instead of about 100 MB).
Classes: cat, dog, house, tree, apple, car, clock, star, fish, flower. `data/raw/manifest.json` records
exactly what was downloaded.

**Training/inference consistency.** Canvas drawings and dataset bitmaps look different (size, line
width, position, colours). To keep them consistent, the dataset images are passed through the *same*
crop / centre / scale code as canvas drawings, and both use the same normalisation, `(x/255 - 0.5)/0.5`.

## Setup (macOS, Apple Silicon)

Use a folder that is **not** inside iCloud Drive (for example `~/Projects/chimera`), otherwise files
may be evicted or fail to sync.

```bash
cd chimera
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m utils.doctor            # checks the environment and tells you what is missing
```

## Usage

```bash
python -m data.download_dataset   # about 40 MB; safe to re-run (existing files are skipped)
python -m training.train          # default: 8 epochs, batch size 128, learning rate 0.001
python -m training.evaluate       # re-evaluate the saved model on the held-out test split
streamlit run app.py              # opens http://localhost:8501
pytest -v                         # run the test-suite
```

Useful options:

```bash
python -m training.train --epochs 12 --batch-size 64 --lr 0.0005
python -m training.train --epochs 2 --samples-per-class 1000   # quick smoke run
python -m training.train --cpu                                  # skip Apple-GPU (MPS)
python -m data.download_dataset --samples-per-class 10000
python -m models.inference my_drawing.png                       # classify an image file
```

Training uses the Apple-Silicon GPU through PyTorch's MPS backend when available and falls back to the
CPU otherwise. Seeds are fixed (default 42) and the train/validation/test split is reproducible.
Results on MPS can still differ slightly between runs because some GPU operations are not deterministic.

## Results

Evaluated on a held-out test split of 7,500 drawings (750 per class) that the model never trained on,
using the best-validation checkpoint (epoch 8 of 8, default settings, 241,994 parameters):

| Metric | Value |
|---|---|
| Accuracy | **92.47%** |
| Top-3 accuracy | 98.09% |
| Macro F1 | 0.9242 |
| Test loss | 0.2416 |

Per class, `house`, `apple`, `clock` and `car` are recognised best (F1 0.96 to 0.97). The weakest are
`dog` (0.800) and `cat` (0.835). The most common mistakes are `dog` mistaken for `cat` (83 times) and
`cat` for `dog` (72), followed by `flower` mistaken for `tree` (41). Those pairs genuinely look similar
when sketched in a few strokes.

These numbers are for the Quick, Draw! test split. Drawing with a mouse is a different distribution, so
accuracy on your own doodles will vary.

## Project structure

```
chimera/
  app.py                  Streamlit entry point
  requirements.txt  pytest.ini  .gitignore  README.md
  .streamlit/config.toml  dark navy / purple theme
  config/settings.py      classes, paths, image size, normalisation, training defaults
  data/
    download_dataset.py   Range-request downloader + manifest
    dataset.py            loading, stratified split, PyTorch Dataset/DataLoader
    preprocessing.py      any image -> normalised 28x28 model input
  models/
    cnn.py                the CNN
    prediction.py         Prediction container and PredictionTimeline
    inference.py          Predictor, model-info helper, image-file CLI
    experiments.py        Experiment Lab: variations and occlusion map
  training/
    train.py  evaluate.py  metrics.py  checkpoint.py
  ui/
    canvas.py  charts.py  components.py  lab.py
  utils/
    device.py             MPS/CPU selection and seeding
    doctor.py             environment checker
  tests/                  pytest suite
  artifacts/              trained model and metrics (created by training, git-ignored)
```

## Limitations (please read)

- **Not true real-time.** The canvas component reports the drawing when a stroke is *finished*, not while
  the pen is moving. Predictions therefore update after every stroke. If your canvas version feels slow,
  switch off "Predict after every stroke" in the sidebar and use the canvas's own update button.
- **Only 10 classes.** Anything else is forced into one of them (a drawing of a banana will be called
  something). A low confidence value is the model's way of saying it is unsure.
- **Distribution gap.** The model was trained on Quick, Draw! sketches, not on mouse-drawn strokes. Brush
  size matters: very thick or very thin strokes can hurt accuracy. The Experiment Lab shows this effect.
- **Subset of the data.** Only the first 5,000 drawings per class are used, in the order stored in the
  source files (not a random sample of the full dataset).
- **No data augmentation** during training. Adding random shifts, scales and stroke-width changes would
  probably make the model more robust to hand-drawn input.

## Troubleshooting

Run `python -m utils.doctor` first. It checks Python, package versions, the compute device, the data
files and the checkpoint.

| Problem | Cause and fix |
|---|---|
| `No trained model found` in the app | Run `python -m training.train` from the project folder, then reload |
| `FileNotFoundError ... download_dataset` | Run `python -m data.download_dataset` |
| Downloader prints an HTTP/network error | Check your connection and re-run; finished classes are skipped |
| `Weights only load failed ... TorchVersion` | Old checkpoint from an earlier version. Delete `artifacts/chimera_cnn.pt` and retrain |
| `StreamlitAPIException ... asset_dir ... file-backed css` when running `pytest` | Recent canvas releases can only be imported inside a running Streamlit app. The project imports the canvas lazily, so make sure you have the current `ui/canvas.py` |
| `st_canvas() got an unexpected keyword argument` | Canvas versions differ in their arguments. The project filters arguments by the installed version; update `ui/canvas.py`. As a last resort: `pip install streamlit-drawable-canvas==0.9.3` |
| "The drawing canvas could not be loaded" in the app | Read the message in the red box; try `pip install "streamlit-drawable-canvas[image]"` |
| Canvas shows but nothing is predicted | Finish a stroke (lift the pen). With "Predict after every stroke" off, press the canvas's update button |
| Streamlit asks for an email on first start | Press Enter to skip, or create `~/.streamlit/credentials.toml` containing `[general]` and `email = ""` |
| Files show "Error"/cloud icons in Finder | The folder is in iCloud Drive. Move the project to a local folder such as `~/Projects` |
| Training is slow | Use `--samples-per-class 1000 --epochs 2` to test first; confirm `Device: mps` in the output |
| Tests fail after changing `config/settings.py` | The tests assume 10 classes and 28 × 28 images; update them together with the config |

## Credits

- Dataset: [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset), Google, CC BY 4.0.
- Canvas: [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas).
