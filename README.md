# LayerDivider

Create transparent image layers from an illustration using SAM 3 text prompts.
Each detected mask becomes a full-size PNG with its original colors and a
transparent background. The interface displays a segmentation preview, a gallery
of individual layers, and downloadable PNGs.

## Windows setup

Requires Python 3.12, Git, an NVIDIA CUDA GPU, and a Hugging Face account with
access to the gated [facebook/sam3](https://huggingface.co/facebook/sam3)
checkpoint.

1. Clone this repository and open PowerShell in the project folder.
2. Run `./install.ps1` once to create the environment and install dependencies.
   If you use the Python launcher (`py`), run `./install_with_launcher.ps1` instead.
3. Accept Meta's model license on Hugging Face, then authenticate:

   ```powershell
   .\venv\Scripts\hf.exe auth login
   ```

4. Start the app:

   ```powershell
   .\run_gui.ps1
   ```

5. Open the URL printed in the terminal, normally <http://localhost:7860>.

The official `sam3.pt` checkpoint is downloaded into `segment_model` on the first
segmentation. The installers install CUDA-enabled PyTorch separately from the
application dependencies in `requirements.txt`.

## Create layers

1. Upload an image.
2. Enter short noun phrases, one per line, such as `hair`, `face`, `shirt`, and
   `shoe`. All prompts are processed together.
3. Click **Segment with SAM 3** and review the segmentation preview.
4. Click **Create Layers** to display individual layers in the gallery and
   download transparent PNGs. Layer filenames include their source prompt.

Generated PNGs are saved in a new `output/sam3_layers_*` folder for each export.

## SAM 3 settings

* **Confidence threshold:** Minimum concept-detection confidence. Lower values
  return more masks; higher values retain more confident matches.
* **Mask threshold:** Probability cutoff used to turn each predicted soft mask
  into pixels.
* **Maximum masks:** Maximum number of the highest-scoring masks retained across
  all prompts.
* **Minimum mask area:** Discard masks smaller than this number of pixels.

## WebUI extension

The extension registers a **LayerDivider** tab with the same SAM 3 interface.
Its host environment must provide Python 3.12, CUDA-enabled PyTorch 2.7 or newer,
and access to the SAM 3 checkpoint.

## Google Colab

Open `layerdivider_launch.ipynb`, select a GPU runtime with Python 3.12 or newer,
and run the setup cells. Authenticate with Hugging Face when prompted, then open
the public Gradio URL printed by the final cell.
