from pathlib import Path

import launch


requirements = Path(__file__).resolve().with_name("requirements.txt")
launch.run_pip(
    f'install -r "{requirements}"',
    "Installing LayerDivider SAM 3 dependencies...",
)
