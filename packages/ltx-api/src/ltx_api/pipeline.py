import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import av
import torch
from tqdm import tqdm

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput as ArgsImageConditioningInput

from ltx_api.config import Settings
from ltx_api.schemas import GenerateRequest, ImageConditioningInput

logger = logging.getLogger(__name__)


class PipelineManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline: DistilledPipeline | None = None
        self._device_str: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device_str(self) -> str:
        return self._device_str

    def load(self) -> None:
        loras = self._build_loras()
        quantization = self._settings.build_quantization_policy()

        logger.info(
            "Loading DistilledPipeline: checkpoint=%s, gemma=%s, upsampler=%s, loras=%s, quantization=%s",
            self._settings.distilled_checkpoint_path,
            self._settings.gemma_root,
            self._settings.spatial_upsampler_path,
            len(loras),
            quantization,
        )

        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=self._settings.distilled_checkpoint_path,
            gemma_root=self._settings.gemma_root,
            spatial_upsampler_path=self._settings.spatial_upsampler_path,
            loras=loras,
            quantization=quantization,
        )

        logger.info("Pipeline loaded successfully")

    def unload(self) -> None:
        self._pipeline = None
        torch.cuda.empty_cache()
        logger.info("Pipeline unloaded, GPU memory cleared")

    def _build_loras(self) -> tuple[LoraPathStrengthAndSDOps, ...]:
        result: list[LoraPathStrengthAndSDOps] = []
        paths = self._settings.lora_paths
        strengths = self._settings.lora_strengths
        for i, path in enumerate(paths):
            strength = strengths[i] if i < len(strengths) else 1.0
            result.append(LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP))
        return tuple(result)

    def _map_image_conditioning(
        self, images: list[ImageConditioningInput]
    ) -> list[ArgsImageConditioningInput]:
        return [
            ArgsImageConditioningInput(
                path=img.path,
                frame_idx=img.frame_idx,
                strength=img.strength,
                crf=img.crf,
            )
            for img in images
            if img.path and img.path != "string" and Path(img.path).exists()
        ]

    def generate(
        self,
        request: GenerateRequest,
        output_path: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> None:
        if self._pipeline is None:
            msg = "Pipeline not loaded. Call load() first."
            raise RuntimeError(msg)

        images = self._map_image_conditioning(request.images)
        tiling_config = TilingConfig.default()

        if progress_callback:
            progress_callback(0.0)

        video_iterator, audio = self._pipeline(
            prompt=request.prompt,
            seed=request.seed,
            height=request.height,
            width=request.width,
            num_frames=request.num_frames,
            frame_rate=request.frame_rate,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=request.enhance_prompt,
        )

        if progress_callback:
            progress_callback(0.5)

        self._encode_video(video_iterator, audio, output_path, request.frame_rate)

        if progress_callback:
            progress_callback(1.0)

    def _encode_video(
        self,
        video: Iterator[torch.Tensor] | torch.Tensor,
        audio: Audio,
        output_path: str,
        fps: float,
    ) -> None:
        if isinstance(video, torch.Tensor):
            video = iter([video])

        first_chunk = next(video)
        _, height, width, _ = first_chunk.shape

        container = av.open(output_path, mode="w")
        try:
            stream = container.add_stream("libx264", rate=int(fps))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}

            audio_stream = container.add_stream("aac", rate=audio.sampling_rate)
            audio_stream.codec_context.sample_rate = audio.sampling_rate
            audio_stream.codec_context.layout = "stereo"

            for video_chunk in tqdm([first_chunk, *video], desc="Encoding frames"):
                video_chunk_cpu = video_chunk.to("cpu").numpy()
                for frame_array in video_chunk_cpu:
                    frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

            self._write_audio(container, audio_stream, audio)
        finally:
            container.close()

        logger.info("Video written to %s", output_path)

    def _write_audio(self, container: av.container.Container, audio_stream: av.audio.AudioStream, audio: Audio) -> None:
        samples = audio.waveform
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.shape[1] != 2 and samples.shape[0] == 2:
            samples = samples.T
        if samples.shape[1] != 2:
            logger.warning("Expected 2 audio channels, got %d; treating as mono", samples.shape[1])
            samples = samples[:, :2] if samples.shape[1] > 2 else samples

        if samples.dtype != torch.int16:
            samples = torch.clip(samples, -1.0, 1.0)
            samples = (samples * 32767.0).to(torch.int16)

        frame_in = av.AudioFrame.from_ndarray(
            samples.contiguous().reshape(1, -1).cpu().numpy(),
            format="s16",
            layout="stereo",
        )
        frame_in.sample_rate = audio.sampling_rate
        frame_in.pts = 0

        resampler = av.audio.resampler.AudioResampler(
            format="fltp",
            layout="stereo",
            rate=audio_stream.codec_context.sample_rate or audio.sampling_rate,
        )

        next_pts = 0
        for rframe in resampler.resample(frame_in):
            if rframe.pts is None:
                rframe.pts = next_pts
            next_pts += rframe.samples
            rframe.sample_rate = frame_in.sample_rate
            container.mux(audio_stream.encode(rframe))

        for packet in audio_stream.encode():
            container.mux(packet)
