
# Change Working Directory
Set-Location $PSScriptRoot

# Check whether Pylauncher is installed
$list = (py --list 2> $null)
if(!$?){
    Write-Error "Pytheon launcher failed."
    Pause
    exit
}

# Check minor version of Python
$requiredMinorVersion = "3\.12"
if(!($list -match $requiredMinorVersion)){
    Write-Error -Message "Python 3.12 is not found. Please install Python 3.12."
    Pause
    exit
}

# Check micro version of Python
$requiredMicroVersion = "3\.12\.\d+"
$pyVersion = py -3.12 --version
if(!($pyVersion -match $requiredMicroVersion)){
    Write-Error "Installed Python is not a supported Python 3.12 release."
    Pause
    exit
}

# Create venv folder
if (!(Test-Path -Path "venv")) {
    Write-Output  "Creating venv for python..."
    py -3.12 -m venv venv
    if(!$?){
        Pause
        exit
    }
}

.\venv\Scripts\activate
if(!$?){
    Pause
    exit
}

# Install SAM 3 compatible CUDA PyTorch first
Write-Output "Installing PyTorch with CUDA 12.8 for SAM 3..."
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
if(!$?){
    Write-Error "CUDA PyTorch installation failed."
    Pause
    exit
}

# Install dependent packages
Write-Output "Installing dependencies, including the official SAM 3 package..."
pip install --upgrade -r requirements.txt
if(!$?){
    Write-Error "`"pip install --upgrade -r requirements.txt
    `" failed."
    Pause
    exit
}

Write-Output "Install completed."
Write-Output "Before first segmentation, accept access at https://huggingface.co/facebook/sam3"
Write-Output "Then authenticate with: .\venv\Scripts\hf.exe auth login"
Pause
