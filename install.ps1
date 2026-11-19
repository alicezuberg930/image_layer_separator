Set-Location $PSScriptRoot

if (!(Test-Path -Path "venv")) {
    Write-Output  "Creating Python 3.12 venv..."
    python3.12 -m venv venv
}
.\venv\Scripts\activate

Write-Output "Installing PyTorch with CUDA 12.8 for SAM 3..."
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Output "Installing dependencies, including the official SAM 3 package..."
pip install --upgrade -r requirements.txt

Write-Output "Install completed"
Write-Output "Before first segmentation, accept access at https://huggingface.co/facebook/sam3"
Write-Output "Then authenticate with: .\venv\Scripts\hf.exe auth login"
Read-Host | Out-Null ;
