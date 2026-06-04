import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse(relative_path: str) -> ast.AST:
  return ast.parse((REPO_ROOT / relative_path).read_text())


def _class_method(module: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
  for node in module.body:
    if isinstance(node, ast.ClassDef) and node.name == class_name:
      for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == method_name:
          return item
  raise AssertionError(f"Could not find {class_name}.{method_name}")


def _has_keyword_call(func: ast.FunctionDef, attr_name: str, keyword_name: str) -> bool:
  for node in ast.walk(func):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr_name:
      if any(keyword.arg == keyword_name for keyword in node.keywords):
        return True
  return False


class InferenceOmicsWiringTest(unittest.TestCase):

  def test_maxengine_prefill_threads_omics_inputs(self):
    module = _parse("src/maxtext/inference/maxengine/maxengine.py")
    maxengine_class = "MaxEngine"

    for method_name in ("_nnx_run_model", "prefill_aot", "_prefill_jit", "prefill"):
      method = _class_method(module, maxengine_class, method_name)
      self.assertIn("omics_inputs", [arg.arg for arg in method.args.args + method.args.kwonlyargs])

    nnx_run_model = _class_method(module, maxengine_class, "_nnx_run_model")
    self.assertTrue(any("omics_inputs" in ast.unparse(node) for node in ast.walk(nnx_run_model) if isinstance(node, ast.Call)))

    prefill_aot = _class_method(module, maxengine_class, "prefill_aot")
    self.assertTrue(_has_keyword_call(prefill_aot, "prefill", "omics_inputs"))

    prefill_jit = _class_method(module, maxengine_class, "_prefill_jit")
    self.assertTrue(_has_keyword_call(prefill_jit, "_nnx_run_model", "omics_inputs"))
    self.assertTrue(_has_keyword_call(prefill_jit, "apply", "omics_inputs"))

    prefill = _class_method(module, maxengine_class, "prefill")
    self.assertTrue(_has_keyword_call(prefill, "_prefill_jit", "omics_inputs"))

  def test_generator_threads_omics_vectors(self):
    module = _parse("benchmarks/api_server/maxtext_generator.py")
    generator_class = "MaxTextGenerator"

    generation_stream = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "GenerationStream")
    stream_fields = [
        item.target.id
        for item in generation_stream.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    ]
    self.assertIn("omics_inputs", stream_fields)

    for method_name in ("generate_batch", "_process_chunk", "_initialize_streams_and_state"):
      method = _class_method(module, generator_class, method_name)
      self.assertIn("omics_vectors", [arg.arg for arg in method.args.args + method.args.kwonlyargs])

    self.assertIsNotNone(_class_method(module, generator_class, "_prepare_omics_input"))

    run_prefill = _class_method(module, generator_class, "_run_prefill_step")
    self.assertTrue(_has_keyword_call(run_prefill, "prefill", "omics_inputs"))

  def test_server_models_expose_omics_vector(self):
    module = _parse("benchmarks/api_server/server_models.py")
    for class_name in ("CompletionRequest", "ChatCompletionRequest"):
      cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name)
      fields = [
          item.target.id
          for item in cls.body
          if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
      ]
      self.assertIn("omics_vector", fields)

  def test_server_batches_omics_vectors(self):
    source = (REPO_ROOT / "benchmarks/api_server/maxtext_server.py").read_text()
    self.assertIn('"omics_vectors": []', source)
    self.assertIn('getattr(req, "omics_vector", None)', source)
    self.assertIn('LLM.generate_batch(prompts=payload["prompts"], **payload["params"])', source)


if __name__ == "__main__":
  unittest.main()
