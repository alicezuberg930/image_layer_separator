import importlib.util
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import psutil
import requests

from layerdivider_runtime import sam3_cache


PROJECT_DIR = Path(__file__).resolve().parent.parent


class GradioReloadTests(unittest.TestCase):
    def test_command_line_help_documents_reload(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "demo.py"), "--help"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        self.assertIn("--reload", result.stdout)

    def test_reimporting_segmentation_code_reuses_the_cached_model(self):
        processor = Mock()
        model = object()
        spec = importlib.util.spec_from_file_location(
            "_reloaded_segment", PROJECT_DIR / "ldivider" / "ld_segment.py"
        )
        reloaded = importlib.util.module_from_spec(spec)
        with (
            patch.object(sam3_cache, "model", model),
            patch.object(sam3_cache, "processor", processor),
            patch.object(sam3_cache, "checkpoint_path", "cached.pt"),
            patch("torch.cuda.is_available", return_value=True),
        ):
            spec.loader.exec_module(reloaded)
            self.assertIs(reloaded.get_sam3_processor(0.7, "unused"), processor)
            self.assertIs(reloaded.sam3_cache.model, model)
            processor.set_confidence_threshold.assert_called_once_with(0.7)

    def test_saving_ui_source_reloads_the_running_server(self):
        with tempfile.TemporaryDirectory(prefix=".reload-check-", dir=PROJECT_DIR) as folder:
            root = Path(folder)
            source_dir = root / "ldivider"
            source_dir.mkdir()
            (root / "scripts").mkdir()
            for source in (PROJECT_DIR / "ldivider").glob("*.py"):
                shutil.copy2(source, source_dir / source.name)
            for source in (PROJECT_DIR / "ldivider").glob("*.js"):
                shutil.copy2(source, source_dir / source.name)
            shutil.copy2(PROJECT_DIR / "layerdivider_runtime.py", root / "layerdivider_runtime.py")

            entrypoint = (PROJECT_DIR / "demo.py").read_text(encoding="utf-8")
            self.assertIn("demo.launch(share=True)", entrypoint)
            (root / "demo.py").write_text(
                entrypoint.replace("demo.launch(share=True)", "demo.launch(share=False)"),
                encoding="utf-8",
            )
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            environment = dict(
                os.environ,
                GRADIO_ANALYTICS_ENABLED="False",
                GRADIO_SERVER_NAME="127.0.0.1",
                GRADIO_SERVER_PORT=str(port),
                GRADIO_NUM_PORTS="1",
            )
            for key in ("GRADIO_WATCH_DIRS", "GRADIO_WATCH_FILE", "GRADIO_WATCH_DEMO_NAME"):
                environment.pop(key, None)
            log_path = root / "server.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(root / "demo.py"), "--reload"],
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                client = requests.Session()
                client.trust_env = False

                def wait_for_button(label):
                    deadline = time.monotonic() + 25
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            break
                        try:
                            response = client.get(
                                f"http://127.0.0.1:{port}/config", timeout=1
                            )
                            response.raise_for_status()
                            config = response.json()
                            if any(
                                item.get("type") == "button"
                                and item["props"].get("value") == label
                                for item in config["components"]
                            ):
                                return config
                        except (requests.RequestException, ValueError):
                            pass
                        time.sleep(0.1)
                    self.fail(
                        f"Button {label!r} was not served.\n"
                        + log_path.read_text(encoding="utf-8", errors="replace")
                    )

                try:
                    wait_for_button("Generate PSD")
                    # Let the initial file timestamp scan finish before editing.
                    time.sleep(0.3)
                    ui_path = source_dir / "ld_ui.py"
                    code = ui_path.read_text(encoding="utf-8")
                    ui_path.write_text(
                        code.replace('value="Generate PSD"', 'value="Generate PSD (reloaded)"'),
                        encoding="utf-8",
                    )
                    timestamp = time.time() + 2
                    os.utime(ui_path, (timestamp, timestamp))
                    wait_for_button("Generate PSD (reloaded)")
                    self.assertIsNone(process.poll())
                    self.assertIn(
                        "Changes detected in:",
                        log_path.read_text(encoding="utf-8", errors="replace"),
                    )
                finally:
                    client.close()
                    try:
                        server = psutil.Process(process.pid)
                        processes = server.children(recursive=True) + [server]
                    except psutil.NoSuchProcess:
                        processes = []
                    for server in processes:
                        try:
                            server.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    _, remaining = psutil.wait_procs(processes, timeout=5)
                    for server in remaining:
                        server.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
