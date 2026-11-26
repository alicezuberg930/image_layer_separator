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

## Development reload

Start Gradio with source reloading enabled:

```powershell
.\run_gui.ps1 -Reload
```

Or, with the virtual environment activated, run Python directly:

```powershell
python demo.py --reload
```

Use `python demo.py` without extra arguments for a normal run.

Saving Python files in `ldivider` or `scripts` rebuilds the Gradio interface and
automatically refreshes the browser. The server keeps running, so its public
share URL stays the same. The virtual environment, model checkpoints, and
generated outputs are excluded from watching.

Reloading rebuilds the UI and discards unsaved editor previews. The current job
is restored from browser storage, including its image, prompts, and saved layers.
The cached SAM 3 model is preserved across source reloads.
Changes to `demo.py`, `layerdivider_runtime.py`, launcher scripts,
installed dependencies, or model initialization require a manual restart.
Run `./run_gui.ps1` without `-Reload` for the normal server.

This reload option applies to the Gradio server. To reload the standalone HTTP
API while developing, use:

```powershell
.\venv\Scripts\python.exe -m uvicorn scripts.api:app --reload --reload-dir scripts --reload-dir ldivider
```

## Create layers

1. Upload an image.
2. Enter short noun phrases, one per line, such as `hair`, `face`, `shirt`, and
   `shoe`. All prompts are processed together.
3. Click **Segment with SAM 3** and review the segmentation preview.
4. Click **Create Layers** to display individual layers in the gallery and
   download transparent PNGs. Layer filenames include their source prompt.
5. Click the **pencil icon** on any layer preview (or its enlarged view). The dialog shows
   the original image, the selected part's preview, its saved part name, and the
   four SAM 3 sliders. The part name is filled in automatically.
6. Adjust the sliders and click **Preview**. Click **Replace layer** to save the
   result, or **Cancel** to leave the saved PNG unchanged. If several objects
   match the same part name, editing chooses the mask with the greatest overlap
   with the selected layer. Only the selected layer is replaced.
7. Click the **trash icon** on a layer preview (or its enlarged view) to permanently
   delete its PNG and saved mask. It is removed from the gallery, downloads, and
   future PSD exports. Refreshing or clicking **Create Layers** keeps it deleted.
8. Click **Generate PSD** to combine the current separated PNGs into a layered PSD.
   Each PNG stays a separate editable layer with its name, original size, and
   transparency. The PSD is available to download and is saved at
   `output/jobs/psd/_<uuid>.psd`. Every export gets a new UUID filename.
   Layers and the merged preview use lossless RLE compression.

Generated PNGs are saved directly in `output/jobs/layers/<uuid>/`. A new UUID is
created only after a successful **Segment with SAM 3** operation. **Create Layers**,
previewing, replacing, and deleting a layer keep the same job ID and folder.
Remaining layers keep their PNG filenames. Generate the PSD again after editing
or deleting a layer to include the changes. Previously exported PSD files remain
unchanged. If all layers are deleted, click **Segment with SAM 3** to start again.

The browser remembers its current job ID in local storage and restores the job
after a page refresh or app restart on the same browser origin. The original
image, per-layer prompts, masks, and slider settings are stored privately under
`output/jobs/state/<uuid>/`. Editing always runs on that job's original image,
even if a different image has since been uploaded into the input widget.
Starting a new segmentation preserves previous job folders.

## SAM 3 settings

* **Confidence threshold:** Minimum concept-detection confidence. Lower values
  return more masks; higher values retain more confident matches.
* **Mask threshold:** Probability cutoff used to turn each predicted soft mask
  into pixels.
* **Maximum masks:** Maximum number of the highest-scoring masks retained across
  all prompts.
* **Minimum mask area:** Discard masks smaller than this number of pixels.

## HTTP API without Gradio

Start the standalone API from the project folder:

```powershell
.\venv\Scripts\python.exe .\scripts\api.py
```

The API listens on `http://127.0.0.1:8000`. Interactive documentation is available
at `http://127.0.0.1:8000/docs`. You can choose another address or port with
`--host` and `--port`, or run `python -m uvicorn scripts.api:app` using the venv.
Use one worker so requests share a single cached SAM 3 model.

### POST /segment

Send `multipart/form-data` with an `image` upload and optional `text_prompt`,
`confidence_threshold`, `mask_threshold`, `max_masks`, and `min_mask_area` fields.
These settings use the same defaults as the Gradio interface. The response has a
`segmentation_id` and an `images` array containing the combined segmentation
preview's absolute PNG URL.

```powershell
$segmentation = curl.exe -sS http://127.0.0.1:8000/segment `
    -F "image=@C:\images\illustration.png" `
    -F "text_prompt=hair,face,shirt" | ConvertFrom-Json
$segmentation.images
```

### POST /create-layer

Send JSON containing the `segmentation_id` from `/segment`. No second upload or
model inference is needed. The response includes the same ID and an `images`
array containing one absolute transparent PNG URL per retained mask.

```powershell
$body = @{ segmentation_id = $segmentation.segmentation_id } | ConvertTo-Json
$layers = Invoke-RestMethod http://127.0.0.1:8000/create-layer `
    -Method Post -ContentType "application/json" -Body $body
$layers.images
```

### POST or PATCH /edit-layer

Send the job ID and the stable PNG `filename` from a `/create-layer` URL. The
route reruns SAM 3 with that layer's saved prompt, chooses the new mask with the
greatest overlap, and replaces only that PNG. Any omitted setting keeps its
previous value.

```powershell
$filename = [IO.Path]::GetFileName(([uri]$layers.images[0]).AbsolutePath)
$edit = @{
    segmentation_id = $segmentation.segmentation_id
    filename = $filename
    mask_threshold = 0.4
    min_mask_area = 25
} | ConvertTo-Json
$layers = Invoke-RestMethod http://127.0.0.1:8000/edit-layer `
    -Method Patch -ContentType "application/json" -Body $edit
```

The response uses the same shape as `/create-layer` and contains all current
layer URLs. The edited layer keeps its URL and filename.

### DELETE or POST /delete-layer

Send `segmentation_id` and `filename` as JSON. The PNG and its saved mask are
permanently removed, and the response contains the remaining layer URLs.
Repeating the request for the same filename is safe.

```powershell
$delete = @{
    segmentation_id = $segmentation.segmentation_id
    filename = $filename
} | ConvertTo-Json
$layers = Invoke-RestMethod http://127.0.0.1:8000/delete-layer `
    -Method Delete -ContentType "application/json" -Body $delete
```

### POST /create-psd

Send `segmentation_id` after creating the PNG layers. A new PSD is generated
from the current layers, including replacements and excluding deletions. The
`psd` response property is an absolute download URL; each request creates a new
file.

```powershell
$psd = Invoke-RestMethod http://127.0.0.1:8000/create-psd `
    -Method Post -ContentType "application/json" -Body $body
$psd.psd
```

The API uses the same job layout as Gradio. `segmentation_id` is the job UUID;
repeating `/create-layer` reuses `output/jobs/layers/<segmentation_id>/` and its
stable image URLs. IDs remain usable after restarting the API as long as
`output/jobs/state` is retained. Only generated artifacts are served: `/images/`
maps to `output/jobs/layers`, `/previews/` maps to `output/jobs/previews`, and
`/psd/` maps to generated PSD exports. Original images and mask/settings files
are not served. Image endpoints return JSON in this shape (`/create-layer`,
`/edit-layer`, and `/delete-layer` return the individual layer URLs in `images`):

```json
{
  "segmentation_id": "0123456789abcdef0123456789abcdef",
  "images": ["http://127.0.0.1:8000/previews/0123456789abcdef0123456789abcdef/segmentation.png"]
}
```

Invalid uploads return HTTP 400, invalid settings, filenames, or unusable masks
return 422, unknown segmentation IDs return 404, and an unavailable SAM 3 model
returns 503.

## WebUI extension

The extension registers a **LayerDivider** tab with the same SAM 3 interface.
Its host environment must provide Python 3.12, CUDA-enabled PyTorch 2.7 or newer,
and access to the SAM 3 checkpoint.

## Google Colab

Open `layerdivider_launch.ipynb`, select a GPU runtime with Python 3.12 or newer,
and run the setup cells. Authenticate with Hugging Face when prompted, then open
the public Gradio URL printed by the final cell.
