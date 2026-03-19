# Firefly Services Nodes for ComfyUI

Custom nodes for Adobe Firefly, Photoshop, Substance 3D, InDesign, and Video Reframe APIs — all within ComfyUI.

## What's Included

| Module | Nodes | Description |
|--------|-------|-------------|
| **Firefly** | 8 | Text to Image, Text to Video, Generative Fill, Generative Expand, Generate Similar, Object Composite, Upload Image, List Custom Models |
| **Photoshop** | 14 | Load PSD, Smart Object, Document Operations, Rendition, PSD Manifest, Text Edit, Remove Background, Depth Blur, Product Crop, Photo Restoration, ActionJSON, Mask Body Parts, Mask Objects, Easy Nodes |
| **Substance 3D** | 7 | Render Basic, Render, Composite, Convert, Assemble Scene, Describe, Load Files |
| **InDesign** | 2 | Load Files, Data Merge |
| **Video Reframe** | 1 | Video overlay compositing with timing/position controls |
| **Utils** | 7 | File Upload to S3, Combine Masks, Cutout Mask, Cutout Alpha Fade, Filter Masks, Mask Morphology, Outline Mask |

**Total: 39 nodes**

Also includes 11 example workflows and 13 example input assets that auto-copy to your ComfyUI input folder.

---

## Prerequisites

- **ComfyUI** — [Download ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- **ComfyUI Manager** — Required for easy installation. Install it first:
  ```bash
  cd ComfyUI/custom_nodes
  git clone https://github.com/Comfy-Org/ComfyUI-Manager.git
  ```
  Then restart ComfyUI. You'll see a "Manager" button in the top menu.

- **Adobe API Credentials** — You need a Firefly Services API client ID and secret from [Adobe Developer Console](https://developer.adobe.com/console)

- **AWS S3 Bucket** — Required for Photoshop, Substance 3D, InDesign, and Video Reframe nodes (used for file transfer with Adobe APIs)

---

## Installation

### Option 1: ComfyUI Manager (Recommended)

1. Open ComfyUI in your browser
2. Click **Manager** in the top menu
3. Click **Install via Git URL**
4. Paste: `https://github.com/crehop/firefly_services_nodes`
5. Click Install and restart ComfyUI

### Option 2: Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/crehop/firefly_services_nodes.git
cd firefly_services_nodes
pip install -r requirements.txt
```

Restart ComfyUI.

---

## Configuration

On first run, a `firefly_config.json` file is automatically created in the node package folder with empty values. Fill in your credentials:

```json
{
  "client_id": "YOUR_ADOBE_CLIENT_ID",
  "client_secret": "YOUR_ADOBE_CLIENT_SECRET",
  "aws_access_key_id": "YOUR_AWS_ACCESS_KEY",
  "aws_secret_access_key": "YOUR_AWS_SECRET_KEY",
  "aws_region": "us-east-1",
  "aws_bucket": "YOUR_BUCKET_NAME"
}
```

The file is located at: `ComfyUI/custom_nodes/firefly_services_nodes/firefly_config.json`

### Where to get credentials

- **Adobe API**: Go to [Adobe Developer Console](https://developer.adobe.com/console), create a project, and add the Firefly Services API. Copy the Client ID and Client Secret.
- **AWS S3**: Create an S3 bucket in AWS, then create an IAM user with S3 read/write access. Copy the Access Key ID and Secret Access Key.

---

## Example Workflows

The `example_workflows/` folder contains 11 ready-to-use hackathon workflows:

| Workflow | Description |
|----------|-------------|
| **Day 1A** - Generate Mascot | Generate a mascot with Firefly Text to Image + Remove Background |
| **Day 1B** - Edit Can Texture | Replace smart object + edit text/colors in a PSD template |
| **Day 1** - Full Flow | Combined: generate mascot → edit PSD → 3 flavor variants |
| **Day 2** | Additional workflow |
| **Day 3A** - Generate Video | Generate video from a can image with Firefly Text to Video |
| **Day 3B** - Video Reframe | Apply localized overlay to a video |
| **Day 3C** - Data Merge | InDesign data merge with CSV + template |
| **Day 3** - Ad Production | Full ad production pipeline |
| **Localization Variants** | Generate localized variants (US/UK/DE/JP) |
| **Full Pipeline** | Combined Day 1 + Day 1B pipeline |
| **COMPLETE End to End** | Full end-to-end production workflow |

### Loading a workflow

1. In ComfyUI, click **Load** (or drag and drop)
2. Navigate to `ComfyUI/custom_nodes/firefly_services_nodes/example_workflows/`
3. Select a workflow JSON file

### Example input files

Input assets (PSD templates, images, CSV data) are automatically copied to your ComfyUI `input/` folder on first load. No manual setup needed.

---

## Node Categories

All nodes appear in ComfyUI's node menu under:

- `api node/firefly` — Firefly image and video generation
- `api node/photoshop` — Photoshop image editing and PSD operations
- `api node/Substance 3D` — 3D rendering and compositing
- `api node/InDesign` — InDesign data merge
- `api node/video` — Video reframe and overlay
- `api node/Firefly Utils` — Mask utilities and file upload

---

## Updating

### Via ComfyUI Manager
Click **Manager** → **Update All** → Restart ComfyUI

### Manual
```bash
cd ComfyUI/custom_nodes/firefly_services_nodes
git pull
pip install -r requirements.txt
```

---

## Troubleshooting

**Nodes not appearing?**
- Restart ComfyUI after installation
- Check the ComfyUI terminal/console for error messages
- Verify `firefly_config.json` exists (auto-created on first run)

**API errors (401/403)?**
- Check your `client_id` and `client_secret` in `firefly_config.json`
- Verify your Adobe API credentials have the correct scopes

**S3 upload errors?**
- Check your AWS credentials in `firefly_config.json`
- Verify your S3 bucket exists and the IAM user has read/write access

**Missing input files?**
- Input assets auto-copy on first load. If missing, manually copy from `example_inputs/` to your ComfyUI `input/` folder

---

## License

MIT
