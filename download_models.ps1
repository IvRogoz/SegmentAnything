$ErrorActionPreference = "Stop"

$models = @(
    @{
        Path = "sam2.1_l.pt"
        Url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_l.pt"
    },
    @{
        Path = "sam2.1_s.pt"
        Url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_s.pt"
    },
    @{
        Path = "sam2.1_t.pt"
        Url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2.1_t.pt"
    },
    @{
        Path = "sam2_t.pt"
        Url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam2_t.pt"
    },
    @{
        Path = "sam_b.pt"
        Url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/sam_b.pt"
    },
    @{
        Path = "static/models/depth_anything_v2_vits.onnx"
        Url = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx"
    },
    @{
        Path = "static/models/midas_v21_small_256.onnx"
        Url = "https://huggingface.co/julienkay/sentis-MiDaS/resolve/main/onnx/midas_v21_small_256.onnx"
    },
    @{
        Path = "vendor/EdgeTAM/checkpoints/edgetam.pt"
        Url = "https://raw.githubusercontent.com/facebookresearch/EdgeTAM/main/checkpoints/edgetam.pt"
    }
)

foreach ($model in $models) {
    $destination = Join-Path $PSScriptRoot $model.Path
    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        Write-Host "Already present: $($model.Path)"
        continue
    }

    $directory = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $temporary = "$destination.part"

    Write-Host "Downloading: $($model.Path)"
    try {
        Invoke-WebRequest -Uri $model.Url -OutFile $temporary -UseBasicParsing
        Move-Item -LiteralPath $temporary -Destination $destination -Force
    }
    catch {
        Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
        throw
    }
}

Write-Host "All model files are present."
