# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OpenAI-compatible video API encoding helpers."""

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.postprocess import rife_interpolator
from vllm_omni.entrypoints.openai import video_api_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _install_fake_video_mux(monkeypatch, mux_calls):
    def _fake_mux_video_audio_bytes(frames, audio, fps, audio_sample_rate, video_codec_options=None):
        mux_calls.append(
            {
                "frames": frames,
                "audio": audio,
                "fps": fps,
                "audio_sample_rate": audio_sample_rate,
                "video_codec_options": video_codec_options,
            }
        )
        return b"fake-video"

    monkeypatch.setattr(
        "vllm_omni.diffusion.utils.media_utils.mux_video_audio_bytes",
        _fake_mux_video_audio_bytes,
    )


def _install_fake_export_to_video(monkeypatch, export_calls):
    def _fake_export_to_video(frames, output_path, fps):
        export_calls.append(
            {
                "frames": frames,
                "output_path": output_path,
                "fps": fps,
            }
        )
        with open(output_path, "wb") as f:
            f.write(b"fake-video")

    monkeypatch.setattr("diffusers.utils.export_to_video", _fake_export_to_video)


def test_encode_video_bytes_exports_frames_without_interpolation(monkeypatch):
    export_calls = []
    _install_fake_export_to_video(monkeypatch, export_calls)

    frames = [np.full((2, 2, 3), fill_value=i / 5, dtype=np.float32) for i in range(5)]
    video_bytes = video_api_utils._encode_video_bytes(
        frames,
        fps=8,
    )

    assert video_bytes == b"fake-video"
    assert len(export_calls) == 1
    assert len(export_calls[0]["frames"]) == 5
    assert export_calls[0]["frames"][0].shape == (2, 2, 3)
    assert export_calls[0]["fps"] == 8


def test_encode_video_bytes_uses_array_fast_path(monkeypatch):
    export_calls = []
    _install_fake_export_to_video(monkeypatch, export_calls)
    monkeypatch.setattr(
        video_api_utils,
        "_normalize_frames",
        lambda frames: (_ for _ in ()).throw(AssertionError("ndarray input should not go through frame list path")),
    )

    video = np.linspace(0.0, 1.0, num=5 * 2 * 2 * 3, dtype=np.float32).reshape(5, 2, 2, 3)
    video_bytes = video_api_utils._encode_video_bytes(video, fps=12)

    assert video_bytes == b"fake-video"
    assert len(export_calls) == 1
    assert len(export_calls[0]["frames"]) == 5
    assert export_calls[0]["frames"][0].flags["C_CONTIGUOUS"]
    assert export_calls[0]["fps"] == 12.0


def test_encode_video_bytes_without_audio_uses_diffusers_export(monkeypatch):
    export_calls = []
    _install_fake_export_to_video(monkeypatch, export_calls)
    monkeypatch.setattr(
        "vllm_omni.diffusion.utils.media_utils.mux_video_audio_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("no-audio path should not use mux_video_audio_bytes")
        ),
    )

    video = np.linspace(0.0, 1.0, num=4 * 2 * 2 * 3, dtype=np.float32).reshape(4, 2, 2, 3)
    video_bytes = video_api_utils._encode_video_bytes(video, fps=10)

    assert video_bytes == b"fake-video"
    assert len(export_calls) == 1
    assert export_calls[0]["fps"] == 10
    assert len(export_calls[0]["frames"]) == 4


def test_rife_model_inference_runs_on_dummy_tensors():
    model = rife_interpolator.Model().eval()
    img0 = torch.rand(1, 3, 32, 32)
    img1 = torch.rand(1, 3, 32, 32)

    output = model.inference(img0, img1, scale=1.0)

    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()


def test_frame_interpolator_runs_actual_torch_tensor_path(monkeypatch):
    model = rife_interpolator.Model().eval()
    interpolator = rife_interpolator.FrameInterpolator()
    monkeypatch.setattr(interpolator, "_ensure_model_loaded", lambda preferred_device=None: model)

    video = torch.zeros(1, 3, 2, 32, 32)
    output_video, multiplier = interpolator.interpolate_tensor(video, exp=1, scale=1.0)

    assert multiplier == 2
    assert output_video.shape == (1, 3, 3, 32, 32)
    assert torch.isfinite(output_video).all()


def test_frame_interpolator_uses_platform_device_when_tensor_is_cpu(monkeypatch):
    chosen_devices = []
    model = rife_interpolator.Model().eval()

    def _fake_ensure_model_loaded(*, preferred_device=None):
        chosen_devices.append(preferred_device)
        return model

    interpolator = rife_interpolator.FrameInterpolator()
    monkeypatch.setattr(interpolator, "_ensure_model_loaded", _fake_ensure_model_loaded)
    monkeypatch.setattr(model.flownet, "to", lambda device: model.flownet)
    monkeypatch.setattr(rife_interpolator, "_select_torch_device", lambda: torch.device("cuda"))

    video = torch.zeros(1, 3, 2, 32, 32)
    output_video, multiplier = interpolator.interpolate_tensor(video, exp=1, scale=1.0)

    assert chosen_devices == [torch.device("cuda")]
    assert multiplier == 2
    assert output_video.shape == (1, 3, 3, 32, 32)
