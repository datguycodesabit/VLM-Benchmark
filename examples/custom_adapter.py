"""Example Python adapter; replace the checkpoint/model code for your local OCR model.

From a checkout with the TrOCR extra installed:
  vlm-bench run --prepared data/prepared \
    --models python:examples.custom_adapter:Factory --repeats 1

Model downloads require network access. This example has not been live-validated.
"""

from vlm_bench.adapters import BackendCapabilities


class Factory:
    adapter_version = "1"
    checkpoint = "microsoft/trocr-base-handwritten"
    capabilities = BackendCapabilities(
        supported_tracks=frozenset({"prose"}),
        supported_sample_types=frozenset({"word", "line", "prose"}),
    )

    def identity(self, model, *, settings, runtime):
        return {
            "checkpoint": self.checkpoint,
            "revision": settings.get("revision", "main"),
            "precision": "float32",
            "task": "handwritten-prose",
        }

    def load(self, model, *, device, settings):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.device = device
        self.settings = dict(settings)
        revision = settings.get("revision", "main")
        self.processor = TrOCRProcessor.from_pretrained(self.checkpoint, revision=revision)
        self.model = (
            VisionEncoderDecoderModel.from_pretrained(self.checkpoint, revision=revision)
            .to(device)
            .eval()
        )

    def preprocess(self, image, *, track, prompt, controls):
        return self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

    def infer(self, prepared, *, controls):
        with self.torch.inference_mode():
            return self.model.generate(
                prepared,
                max_new_tokens=controls.get("num_predict", 4096),
                num_beams=self.settings.get("num_beams", 4),
            )

    def decode(self, output):
        return self.processor.batch_decode(output, skip_special_tokens=True)[0]

    def unload(self, model):
        self.model = None
        self.processor = None
