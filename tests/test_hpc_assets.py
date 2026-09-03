import unittest
from pathlib import Path


class HPCAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_definition_pins_python_ai2thor_and_vulkan(self):
        definition = (self.root / "hpc" / "openprop-ai2thor.def").read_text()
        requirements = (self.root / "hpc" / "requirements-ai2thor.txt").read_text()
        self.assertIn("ubuntu:24.04", definition)
        self.assertIn("python3-venv", definition)
        self.assertIn("libvulkan1", definition)
        self.assertIn("vulkan-tools", definition)
        self.assertIn("nvidia_icd.json", definition)
        self.assertIn("10_nvidia.json", definition)
        self.assertIn("ai2thor==5.0.0", requirements)
        self.assertNotIn("ollama", definition.lower())

    def test_slurm_uses_one_gpu_persistent_cache_and_real_preflight(self):
        slurm = (self.root / "hpc" / "ai2thor_capture.slurm").read_text()
        self.assertIn("#SBATCH --gres=gpu:1", slurm)
        self.assertIn('"$CONTAINER_RUNTIME" exec --nv', slurm)
        self.assertIn("HOST_NVIDIA_ICD", slurm)
        self.assertIn("select a Vulkan-capable node", slurm)
        self.assertIn("VK_ICD_FILENAMES", slurm)
        self.assertIn("openprop-ai2thor-v3.sif", slurm)
        self.assertIn('$OPENPROP_CACHE/ai2thor:$HOME/.ai2thor', slurm)
        self.assertNotIn("/root/.ai2thor", slurm)
        self.assertIn("preflight_ai2thor.py", slurm)
        self.assertIn("verify_ai2thor_capture.py", slurm)
        self.assertIn("prepare_ai2thor_capture.py", slurm)
        self.assertIn("$SCENE.verification-report.json", slurm)
        self.assertIn("/output/prepared/$SCENE", slurm)
        self.assertNotIn("OPENAI_API_KEY=", slurm)
        self.assertNotIn("mesa-vulkan-drivers", definition := (
            self.root / "hpc" / "openprop-ai2thor.def"
        ).read_text())

    def test_uploaded_bundle_installer_verifies_before_install(self):
        installer = (self.root / "hpc" / "install_uploaded_bundle.sh").read_text()
        verify_at = installer.index("build_hpc_transfer_manifest.py")
        install_at = installer.index("install -m 0444")
        extract_at = installer.index("tar -xzf")
        self.assertLess(verify_at, install_at)
        self.assertLess(verify_at, extract_at)
        self.assertIn('PYTHON_BIN="${PYTHON_BIN:-python3}"', installer)
        self.assertIn('"$PYTHON_BIN" scripts/build_hpc_transfer_manifest.py', installer)
        self.assertIn('"$PYTHON_BIN" scripts/prepare_visual_datasets.py', installer)
        self.assertIn("refusing to overwrite a different SIF", installer)
        self.assertIn("--require-ready ai2thor_ithor", installer)

    def test_prepare_job_is_cpu_only_and_separates_inputs_from_truth(self):
        slurm = (self.root / "hpc" / "ai2thor_prepare.slurm").read_text()
        self.assertIn("#SBATCH --partition=normal", slurm)
        self.assertNotIn("#SBATCH --gres", slurm)
        self.assertNotIn("--nv", slurm)
        self.assertIn("verify_ai2thor_capture.py", slurm)
        self.assertIn("prepare_ai2thor_capture.py", slurm)
        self.assertIn("/output/prepared/$SCENE", slurm)


if __name__ == "__main__":
    unittest.main()
