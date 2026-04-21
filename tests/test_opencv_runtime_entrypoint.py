from __future__ import annotations

import ast
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "visualizer" / "opencv_runtime.py"


class OpenCvRuntimeEntrypointTests(unittest.TestCase):
    def test_module_imports_vm_base_url_from_app_config(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

        has_vm_base_url_import = False
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "app.config":
                continue
            if any(alias.name == "VM_BASE_URL" for alias in node.names):
                has_vm_base_url_import = True
                break

        self.assertTrue(
            has_vm_base_url_import,
            "opencv_runtime.py main() should import VM_BASE_URL from app.config",
        )

    def test_module_has_main_guard(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

        has_main_guard = False
        for node in tree.body:
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not isinstance(test, ast.Compare):
                continue
            if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
                continue
            if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
                continue
            if len(test.comparators) != 1:
                continue
            comparator = test.comparators[0]
            if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
                has_main_guard = True
                break

        self.assertTrue(
            has_main_guard,
            "opencv_runtime.py should be directly executable via a __main__ guard",
        )


if __name__ == "__main__":
    unittest.main()
