Warning: truncated output (original token count: 26958)
Total output lines: 2893

import os
import shutil
import html
import copy
import re
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

# Offline WebUI keeps the CUDA Graph implementation available, but remains
# eager by default. Set RVC_OFFLINE_CUDA_GRAPH=1 to opt in for benchmarking or
# controlled deployments.
_offline_cuda_graph = os.environ.get("RVC_OFFLINE_CUDA_GRAPH", "0") == "1"
os.environ["RVC_CUDA_GRAPH"] = "1" if _offline_cuda_graph else "0"

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("no_proxy", "localhost, 127.0.0.1, ::1")
os.environ.setdefault("weight_root", "assets/weights")
os.environ.setdefault("weight_pymss_root", "assets/pymss_weights")
os.environ.setdefault("index_root", "logs")
os.environ.setdefault("outside_index_root", "assets/indices")
os.environ.setdefault("rmvpe_root", "assets/rmvpe")

now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
for name in os.listdir(tmp):
    if name == "jieba.cache":
        continue
    path = os.path.join(tmp, name)
    delete = (
        os.remove if os.path.isfile(path) or os.path.islink(path) else shutil.rmtree
    )
    try:
        delete(path)
    except Exception as error:
        print(str(error))

from configs.config import Config, GPU_INDEX, GPU_INFOS, GPU_MEMORY, IS_GPU
from infer.vc.modules import VC
from infer.vc.utils import get_index_path_from_model
from tools.pymss_webui import (
    PYMSS_MODEL_CHOICES,
    get_model_info,
    pymss_separate as _pymss_separate_core,
    stop_pymss_separation as _stop_pymss_separation_core,
)
from tools.file_io import read_text
from tools.process_utils import kill_process_tree
from tools.multispeaker import (
    ManifestError,
    build_manifest_from_root,
    build_manifest_from_rows,
    load_manifest,
    write_manifest,
)
from train.process_ckpt import (
    change_info,
    extract_small_model,
    merge,
    show_info,
)
from i18n.i18n import (
    I18nAuto,
    get_preferred_language,
    save_preferred_language,
)
import torch, platform
import numpy as np
import gradio as gr
import pathlib
import json
from time import sleep
from subprocess import Popen
from random import shuffle
import traceback
import threading
import logging
import socket
import subprocess
import time


logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def find_available_port(start_port, host="0.0.0.0"):
    """Return the first bindable TCP port at or above ``start_port``."""
    if not 1 <= start_port <= 65535:
        raise ValueError(f"Port must be between 1 and 65535, got {start_port}.")

    for port in range(start_port, 65536):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
            return port
        except OSError:
            continue

    raise OSError(
        f"No available TCP port from {start_port} through 65535; WebUI was not started."
    )


def is_gradio_port_in_use_error(error, port):
    """Recognize Gradio's explicit-port conflict without hiding other launch errors."""
    return str(error).startswith(f"Port {port} is in use.")


def launch_webui_with_port_fallback(app, config):
    """Launch Gradio, increasing the requested port until startup succeeds."""
    next_port = config.listen_port
    queued_app = app.queue(concurrency_count=511, max_size=1022)
    while True:
        config.listen_port = find_available_port(next_port)
        if config.listen_port != next_port:
            logger.warning(
                "Port %s is occupied; trying port %s instead.",
                next_port,
                config.listen_port,
            )
        try:
            queued_app.launch(
                server_name="0.0.0.0",
                inbrowser=not config.noautoopen,
                server_port=config.listen_port,
                quiet=True,
            )
            return config.listen_port
        except OSError as error:
            if not is_gradio_port_in_use_error(error, config.listen_port):
                raise
            if config.listen_port == 65535:
                raise OSError(
                    "No available TCP port through 65535; WebUI was not started."
                ) from error
            logger.warning(
                "Port %s became occupied while Gradio was starting; trying the next port.",
                config.listen_port,
            )
            next_port = config.listen_port + 1

runtime_dirs = (
    os.path.join(now_dir, "logs"),
    os.environ["weight_root"],
    os.environ["weight_pymss_root"],
    os.environ["index_root"],
    os.environ["outside_index_root"],
    os.environ["rmvpe_root"],
    os.path.join(now_dir, "assets", "hubert_base"),
    os.path.join(now_dir, "assets", "pretrained"),
    os.path.join(now_dir, "assets", "pretrained_v2"),
)
for runtime_dir in runtime_dirs:
    os.makedirs(runtime_dir, exist_ok=True)
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
vc = VC(config)


i18n = I18nAuto()
logger.info(i18n)
print(
    i18n("当前设备：%s | 推理精度：%s") % (config.device, config.dtype),
    flush=True,
)
# GPU filtering and precision rules are shared with inference/extraction/training.
gpu_infos = list(GPU_INFOS)
gpu_indices = sorted(GPU_INDEX)
if_gpu_ok = IS_GPU
if if_gpu_ok:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = max(1, int(min(GPU_MEMORY[i] for i in gpu_indices)) // 2)
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join(str(i) for i in gpu_indices)
feature_gpus = "%s-%s" % (gpus, gpus) if gpus else ""
default_training_f0_method = "rmvpe" if IS_GPU else "pm"


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


weight_root = os.getenv("weight_root")
weight_pymss_root = os.getenv("weight_pymss_root")
outside_index_root = os.getenv("outside_index_root")

def weight_names():
    return sorted(
        name for name in os.listdir(weight_root) if name.endswith(".pth")
    )


def refresh_weight_choices(previous_names=None, force=False):
    current_names = tuple(weight_names())
    if force or current_names != previous_names:
        return current_names, change_choices()
    return current_names, {"__type__": "update"}


names = weight_names()
pymss_names = PYMSS_MODEL_CHOICES


def change_choices():
    return {"choices": weight_names(), "__type__": "update"}


def clean():
    return {"value": "", "__type__": "update"}


def selected_speaker_id(slider_value, dropdown_value):
    value = dropdown_value if dropdown_value is not None else slider_value
    if isinstance(value, str):
        match = re.search(r"ID\s*[:：]\s*(\d+)\s*[)）]?\s*$", value)
        if match:
            return int(match.group(1))
    return int(value)


def normalize_index_path(file_index):
    return (
        str(file_index or "")
        .strip(" ")
        .strip('"')
        .strip("\n")
        .strip('"')
        .strip(" ")
        .replace("trained", "added")
    )


def report_missing_index(file_index):
    index_path = normalize_index_path(file_index)
    if index_path and not os.path.isfile(index_path):
        message = i18n("索引文件不存在，将不使用索引继续推理：%s") % index_path
        print(message, flush=True)
        raise gr.Error(message)


def update_speaker_index(model_name, slider_value, dropdown_value):
    try:
        speaker_id = selected_speaker_id(slider_value, dropdown_value)
    except (TypeError, ValueError):
        speaker_id = None
    path = get_index_path_from_model(model_name, speaker_id)
    update = {"value": path, "__type__": "update"}
    return update, dict(update)


def update_dropdown_speaker_index(model_name, dropdown_value):
    return update_speaker_index(model_name, 0, dropdown_value)


def vc_single_with_speaker(slider_value, dropdown_value, *args):
    return vc.vc_single(selected_speaker_id(slider_value, dropdown_value), *args)


def vc_multi_with_speaker(slider_value, dropdown_value, *args):
    yield from vc.vc_multi(selected_speaker_id(slider_value, dropdown_value), *args)


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}

MULTISPEAKER_PAGE_SIZE = 10
MULTISPEAKER_MAX_ROWS = 110


def is_multispeaker_mode(training_mode):
    return training_mode in ("多说话人", i18n("多说话人"))


def manifest_error_text(error):
    text = i18n(error.key)
    return text % error.values if error.values else text


def experiment_path(exp_name):
    return os.path.join(now_dir, "logs", str(exp_name or "").strip())


def prepare_multispeaker_manifest(trainset_dir, exp_name):
    exp_path = experiment_path(exp_name)
    try:
        if str(trainset_dir or "").strip():
            manifest = build_manifest_from_root(trainset_dir)
            write_manifest(exp_path, manifest)
            return manifest
        return load_manifest(exp_path)
    except ManifestError as error:
        raise RuntimeError(manifest_error_text(error))


def load_experiment_manifest(exp_path):
    try:
        return load_manifest(exp_path)
    except ManifestError as error:
        raise RuntimeError(manifest_error_text(error))


def change_training_mode(training_mode):
    multi = is_multispeaker_mode(training_mode)
    label = (
        i18n("多说话人训练集总文件夹路径")
        if multi
        else i18n("输入训练文件夹路径")
    )
    placeholder = i18n("留空则使用辅助页已提交的清单") if multi else ""
    textbox_update = gr.Textbox.update(label=label, placeholder=placeholder)
    if multi:
        textbox_update["value"] = ""
    return textbox_update, gr.Slider.update(visible=not multi)


def sync_exp_name(source_value, target_value):
    if str(source_value or "") == str(target_value or ""):
        return gr.Textbox.update()
    return gr.Textbox.update(value=source_value)


def empty_multispeaker_rows():
    rows = [["", "", "", ""] for _ in range(MULTISPEAKER_MAX_ROWS)]
    rows[0] = ["", "", 0, 1]
    rows[1] = ["", "", 1, 1]
    return rows


def sync_multispeaker_page(rows, active_count, page, values):
    rows = [list(row) for row in (rows or empty_multispeaker_rows())]
    while len(rows) < MULTISPEAKER_MAX_ROWS:
        rows.append(["", "", "", ""])
    start = int(page) * MULTISPEAKER_PAGE_SIZE
    for slot in range(MULTISPEAKER_PAGE_SIZE):
        index = start + slot
        if index >= MULTISPEAKER_MAX_ROWS:
            break
        offset = slot * 4
        rows[index] = [
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]
    return rows


def multispeaker_page_updates(rows, active_count, page):
    active_count = max(2, min(MULTISPEAKER_MAX_ROWS, int(active_count)))
    total_pages = max(1, (active_count + MULTISPEAKER_PAGE_SIZE - 1) // MULTISPEAKER_PAGE_SIZE)
    page = max(0, min(int(page), total_pages - 1))
    start = page * MULTISPEAKER_PAGE_SIZE
    updates = []
    for slot in range(MULTISPEAKER_PAGE_SIZE):
        index = start + slot
        visible = index < active_count
        row = rows[index] if index < len(rows) else ["", "", "", ""]
        updates.extend(
            [
                gr.Textbox.update(value=row[0], visible=visible),
                gr.Textbox.update(value=row[1], visible=visible),
                gr.Number.update(
                    value=row[2] if row[2] not in ("", None) else None,
                    visible=visible,
                ),
                gr.Number.update(
                    value=row[3] if row[3] not in ("", None) else None,
                    visible=visible,
                ),
            ]
        )
    page_text = i18n("第%s/%s页，共%s行") % (page + 1, total_pages, active_count)
    return tuple([rows, active_count, page, page_text] + updates)


def change_multispeaker_page(action, rows, active_count, page, *values):
    rows = sync_multispeaker_page(rows, active_count, page, values)
    active_count = int(active_count)
    page = int(page)
    if action == "add":
        if active_count < MULTISPEAKER_MAX_ROWS:
            used_ids = set()
            for row in rows[:active_count]:
                try:
                    speaker_id = int(float(row[2]))
                except (TypeError, ValueError):
                    continue
                if 0 <= speaker_id <= 109:
                    used_ids.add(speaker_id)
            speaker_id = next(
                value for value in range(110) if value not in used_ids
            )
            rows[active_count] = ["", "", speaker_id, 1]
            active_count += 1
        page = (active_count - 1) // MULTISPEAKER_PAGE_SIZE
    elif action == "remove":
        if active_count > 2:
            active_count -= 1
            rows[active_count] = ["", "", "", ""]
        page = min(page, (active_count - 1) // MULTISPEAKER_PAGE_SIZE)
    elif action == "previous":
        page -= 1
    elif action == "next":
        page += 1
    return multispeaker_page_updates(rows, active_count, page)


def add_multispeaker_row(rows, active_count, page, *values):
    return change_multispeaker_page("add", rows, active_count, page, *values)


def remove_multispeaker_row(rows, active_count, page, *values):
    return change_multispeaker_page("remove", rows, active_count, page, *values)


def previous_multispeaker_page(rows, active_count, page, *values):
    return change_multispeaker_page("previous", rows, active_count, page, *values)


def next_multispeaker_page(rows, active_count, page, *values):
    return change_multispeaker_page("next", rows, active_count, page, *values)


def render_multispeaker_submit_status(message, warning=False):
    color = "#9a6700" if warning else "#116329"
    background = "#fff8c5" if warning else "#dafbe1"
    return (
        '<div class="multispeaker-submit-status" style="border-color:%s;'
        'background:%s;color:%s;">%s</div>'
    ) % (color, background, color, html.escape(message))


def submit_multispeaker_rows(exp_name, rows, active_count, page, *values):
    rows = sync_multispeaker_page(rows, active_count, page, values)
    exp_name = str(exp_name or "").strip()
    if not exp_name:
        raise gr.Error(i18n("实验名不能为空"))
    try:
        manifest, invalid_rows = build_manifest_from_rows(rows[: int(active_count)])
        path = write_manifest(experiment_path(exp_name), manifest)
    except ManifestError as error:
        raise gr.Error(manifest_error_text(error))
    message = i18n("多说话人训练集清单已保存：%s；有效音频%s个") % (
        path,
        len(manifest["entries"]),
    )
    if invalid_rows:
        warning_text = i18n("以下行填写不完整或无效，已忽略：%s") % ", ".join(
            str(index) for index in invalid_rows
        )
        if hasattr(gr, "Warning"):
            gr.Warning(warning_text)
        logger.warning(warning_text)
        message = "%s\n%s" % (warning_text, message)
        return rows, render_multispeaker_submit_status(message, True)
    return rows, render_multispeaker_submit_status(message)


TRAIN_TASK_LOCK = threading.Lock()
TRAIN_TASK = None


def button_update(value=None, variant=None, visible=None):
    update = {"__type__": "update"}
    if value is not None:
        update["value"] = value
    if variant is not None:
        update["variant"] = variant
    if visible is not None:
        update["visible"] = visible
    return update


def render_pymss_progress(percent=0, label="等待开始", state="idle"):
    percent = max(0.0, min(100.0, float(percent or 0)))
    colors = {
        "idle": "#64748b",
        "running": "#2563eb",
        "done": "#15803d",
        "stopped": "#b45309",
        "failed": "#b91c1c",
    }
    color = colors.get(state, colors["running"])
    safe_label = html.escape(str(label or ""))
    return (
        '<div style="min-height:52px;padding:6px 0;">'
        '<div style="display:flex;justify-content:space-between;gap:12px;'
        'align-items:center;margin-bottom:7px;font-size:14px;line-height:20px;">'
        '<span style="overflow-wrap:anywhere;">%s</span>'
        '<strong style="flex:0 0 auto;color:%s;">%.1f%%</strong>'
        "</div>"
        '<div role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        'aria-valuenow="%.1f" style="height:10px;width:100%%;overflow:hidden;'
        'border-radius:4px;background:#e2e8f0;">'
        '<div style="height:100%%;width:%.3f%%;background:%s;"></div>'
        "</div></div>"
    ) % (safe_label, color, percent, percent, percent, color)


def run_pymss_separation(
    model_name,
    inp_root,
    save_root_vocal,
    paths,
    save_root_ins,
    format0,
):
    progress_state = {
        "percent": 0.0,
        "label": "正在准备 PyMSS 分离任务",
        "state": "running",
    }
    busy = False

    def update_progress(event):
        nonlocal busy
        event_type = event.get("event")
        file_count = max(1, int(event.get("file_count") or 1))
        file_index = max(1, int(event.get("file_index") or 1))
        message = str(event.get("message") or "")

        if event_type == "progress":
            done = max(0.0, float(event.get("done") or 0))
            total = max(1.0, float(event.get("total") or 1))
            file_fraction = min(1.0, done / total)
            progress_state["percent"] = (
                (file_index - 1 + file_fraction) / file_count * 100
            )
            progress_state["label"] = "文件 %s/%s · %s · %.0f/%.0f 秒" % (
                file_index,
                file_count,
                message or "正在处理音频",
                done,
                total,
            )
            progress_state["state"] = "running"
        elif event_type == "file_start":
            progress_state["percent"] = (file_index - 1) / file_count * 100
            progress_state["label"] = message
            progress_state["state"] = "running"
        elif event_type == "file":
            progress_state["percent"] = file_index / file_count * 100
            progress_state["label"] = (
                message.splitlines()[0] if message else "文件处理结束"
            )
            progress_state["state"] = "running" if event.get("ok") else "failed"
        elif event_type == "done":
            successful = int(event.get("successful") or 0)
            failed = int(event.get("failed") or 0)
            progress_state["percent"] = 100.0
            progress_state["label"] = "分离完成：成功 %s，失败 %s" % (
                successful,
                failed,
            )
            progress_state["state"] = "done" if failed == 0 else "failed"
        elif event_type == "retry_fp32":
            progress_state["percent"] = 0.0
            progress_state["label"] = message
            progress_state["state"] = "running"
        elif event_type == "cancelled":
            progress_state["label"] = message or "PyMSS 分离任务已停止"
            progress_state["state"] = "stopped"
        elif event_type in {"fatal", "busy"}:
            progress_state["label"] = (
                message.splitlines()[0] if message else "PyMSS 分离任务失败"
            )
            progress_state["state"] = "failed"
            busy = event_type == "busy"
        elif event_type in {"preparing", "precision_attempt", "status"}:
            progress_state["label"] = message
            progress_state["state"] = "running"

    last_info = ""
    try:
        for info in _pymss_separate_core(
            model_name,
            inp_root,
            save_root_vocal,
            paths,
            save_root_ins,
            format0,
            event_callback=update_progress,
        ):
            last_info = info
            start_button = button_update()
            stop_button = button_update()
            if not busy:
                start_button = button_update(visible=False)
                stop_button = button_update(visible=True)
            yield (
                info,
                render_pymss_progress(**progress_state),
                start_button,
                stop_button,
            )
    except Exception:
        last_info = "失败\n%s" % traceback.format_exc()
        progress_state.update(
            {"label": "PyMSS 分离任务失败", "state": "failed"}
        )
        logger.exception("PyMSS WebUI task failed")

    start_button = button_update()
    stop_button = button_update()
    if not busy:
        start_button = button_update(visible=True)
        stop_button = button_update(visible=False)
    yield (last_info, render_pymss_progress(**progress_state), start_button, stop_button)


def stop_pymss_webui():
    return (
        _stop_pymss_separation_core(),
        render_pymss_progress(0, "PyMSS 分离任务已停止", "stopped"),
        button_update(visible=True),
        button_update(visible=False),
    )


def format_status(title, state, detail=""):
    lines = ["【%s】" % i18n(title), "%s：%s" % (i18n("状态"), i18n(state))]
    if detail:
        lines.extend(["", detail.strip()])
    return "\n".join(lines)


def format_workflow_status(step, detail="", completed_steps=None, state="运行中"):
    completed_steps = completed_steps or []
    detail = str(detail).strip()
    lines = []
    if completed_steps:
        lines.append("%s：" % i18n("已完成阶段"))
        lines.extend(
            "✓ %s：%s" % (i18n(completed_step), i18n("已成功"))
            for completed_step in completed_steps
        )
    if step:
        if lines:
            lines.append("")
        lines.append("%s：%s" % (i18n("当前阶段"), i18n(step)))
    if detail:
        lines.extend(["", detail])
    return format_status(
        "一键训练",
        state,
        "\n".join(lines),
    )


def read_log(path, max_lines=40):
    try:
        lines = [line.rstrip() for line in read_text(path, errors="ignore").splitlines()]
        lines = [line for line in lines if line.strip()]
        if len(lines) > max_lines:
            tail_count = max(0, max_lines - 1)
            omitted = len(lines) - tail_count
            tail = lines[-tail_count:] if tail_count else []
            lines = [i18n("……已省略前%s行，仅显示最新状态") % omitted]
            lines.extend(tail)
        return "\n".join(lines)
    except FileNotFoundError:
        return ""


def artifact_names(directory, suffix):
    if not os.path.isdir(directory):
        return set()
    return {
        name.split(".")[0]
        for name in os.listdir(directory)
        if name.lower().endswith(suffix)
    }


def validate_preprocess_outputs(exp_dir):
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    gt_names = artifact_names(os.path.join(exp_path, "0_gt_wavs"), ".wav")
    wav16_names = artifact_names(os.path.join(exp_path, "1_16k_wavs"), ".wav")
    if not gt_names:
        raise RuntimeError(i18n("数据切分没有生成有效训练音频，请检查训练集和数据切分日志"))
    if not wav16_names:
        raise RuntimeError(i18n("数据切分没有生成16k音频，已停止后续特征提取和训练"))
    if not gt_names & wav16_names:
        raise RuntimeError(i18n("数据切分输出文件不匹配，已停止后续特征提取和训练"))


def validate_feature_outputs(exp_dir, version, if_f0):
    exp_path = os.path.join(now_dir, "logs", exp_dir)
    wav16_names = artifact_names(os.path.join(exp_path, "1_16k_wavs"), ".wav")
    feature_name = "3_feature256" if version == "v1" else "3_feature768"
    featu…14958 tokens truncated…("输出信息"),
                            value="",
                            lines=5,
                            max_lines=5,
                            elem_id="training-info-step2a",
                        )
                but1.click(
                    preprocess_dataset,
                    [trainset_dir4, exp_dir1, sr2, np7, training_mode],
                    [info1, but1, stop_but1],
                    api_name="train_preprocess",
                )
                stop_but1.click(
                    stop_preprocess_dataset,
                    [],
                    [info1, but1, stop_but1],
                    queue=False,
                )
                training_mode.change(
                    change_training_mode,
                    [training_mode],
                    [trainset_dir4, spk_id5],
                    queue=False,
                )
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "step2b: 音高与hubert语义特征提取"
                    )
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=320):
                        gpus6 = gr.Textbox(
                            label=i18n(
                                "hubert:以-分隔输入使用的卡号, 例如 0-1-2 使用卡0和卡1和卡2"
                            ),
                            value=feature_gpus,
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                        gpus_rmvpe = gr.Textbox(
                            label=i18n(
                                "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                            ),
                            value=feature_gpus,
                            interactive=True,
                            visible=F0GPUVisible,
                        )
                    with gr.Column(scale=2, min_width=260):
                        gpu_info9 = gr.Textbox(
                            label=i18n("显卡信息"), value=gpu_info, visible=F0GPUVisible
                        )
                        f0method8 = gr.Radio(
                            label=i18n("选择音高提取算法"),
                            choices=["pm", "rmvpe"],
                            value=default_training_f0_method,
                            interactive=True,
                        )
                    with gr.Column(scale=1, min_width=150):
                        but2 = gr.Button(i18n("特征提取"), variant="primary")
                        stop_but2 = gr.Button(
                            i18n("停止特征提取"), variant="stop", visible=False
                        )
                    with gr.Column(
                        scale=3,
                        min_width=280,
                        elem_id="step2b-output-column",
                    ):
                        info2 = gr.Textbox(
                            label=i18n("输出信息"),
                            value="",
                            lines=8,
                            max_lines=8,
                            elem_id="training-info-step2b",
                        )
                f0method8.change(
                    fn=change_f0_method,
                    inputs=[f0method8],
                    outputs=[gpus_rmvpe],
                )
                but2.click(
                    extract_f0_feature,
                    [
                        gpus6,
                        np7,
                        f0method8,
                        if_f0_3,
                        exp_dir1,
                        version19,
                        gpus_rmvpe,
                    ],
                    [info2, but2, stop_but2],
                    api_name="train_extract_f0_feature",
                )
                stop_but2.click(
                    stop_extract_f0_feature,
                    [],
                    [info2, but2, stop_but2],
                    queue=False,
                )
            with gr.Group():
                gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        label=i18n("保存频率save_every_epoch"),
                        value=5,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1200,
                        step=1,
                        label=i18n("总训练轮数total_epoch"),
                        value=20,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label=i18n("每张显卡的batch_size"),
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label=i18n(
                            "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                        ),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label=i18n(
                            "是否在每次保存时间点将最终小模型保存至weights文件夹"
                        ),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("否"),
                        interactive=True,
                    )
                with gr.Row():
                    pretrained_G14 = gr.Textbox(
                        label=i18n("加载预训练底模G路径"),
                        value="assets/pretrained_v2/f0G40k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label=i18n("加载预训练底模D路径"),
                        value="assets/pretrained_v2/f0D40k.pth",
                        interactive=True,
                    )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, sr2],
                    )
                    if_f0_3.change(
                        change_f0,
                        [if_f0_3, sr2, version19],
                        [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
                    )
                    gpus16 = gr.Textbox(
                        label=i18n(
                            "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"
                        ),
                        value=gpus,
                        interactive=True,
                    )
                    but3 = gr.Button(i18n("训练模型"), variant="primary")
                    stop_but3 = gr.Button(
                        i18n("停止训练模型"), variant="stop", visible=False
                    )
                    but4 = gr.Button(i18n("训练特征索引"), variant="primary")
                    stop_but4 = gr.Button(
                        i18n("停止训练索引"), variant="stop", visible=False
                    )
                    but5 = gr.Button(i18n("一键训练"), variant="primary")
                    stop_but5 = gr.Button(
                        i18n("停止一键训练"), variant="stop", visible=False
                    )
                    info3 = gr.Textbox(
                        label=i18n("输出信息"),
                        value="",
                        lines=8,
                        max_lines=8,
                        elem_id="training-info-step3",
                    )
                    but3.click(
                        click_train,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            spk_id5,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                            training_mode,
                        ],
                        [info3, but3, stop_but3, sid0],
                        api_name="train_start",
                    )
                    stop_but3.click(
                        stop_train_model,
                        [],
                        [info3, but3, stop_but3],
                        queue=False,
                    )
                    but4.click(
                        train_index,
                        [exp_dir1, version19, training_mode],
                        [info3, but4, stop_but4],
                    )
                    stop_but4.click(
                        stop_train_index,
                        [],
                        [info3, but4, stop_but4],
                        queue=False,
                    )
                    but5.click(
                        train1key,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            trainset_dir4,
                            spk_id5,
                            np7,
                            f0method8,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                            gpus_rmvpe,
                            training_mode,
                        ],
                        [info3, but5, stop_but5, sid0],
                        api_name="train_start_all",
                    )
                    stop_but5.click(
                        stop_train1key,
                        [],
                        [info3, but5, stop_but5],
                        queue=False,
                    )

        with gr.TabItem(i18n("多说话人训练集辅助")):
            helper_rows_state = gr.State(empty_multispeaker_rows())
            helper_row_count = gr.State(2)
            helper_page = gr.State(0)
            gr.Markdown(
                value=i18n(
                    "为多说话人训练集建立清单。空行会被忽略；路径、说话人名称、说话人ID或重复次数填写不全的行会在提交时提示。"
                )
            )
            with gr.Row():
                helper_exp_name = gr.Textbox(
                    label=i18n("输入实验名，例如：test")
                )
                helper_submit = gr.Button(i18n("提交训练集清单"), variant="primary")
                helper_previous = gr.Button(i18n("上一页"))
                helper_page_label = gr.Markdown(
                    value=i18n("第%s/%s页，共%s行") % (1, 1, 2)
                )
                helper_next = gr.Button(i18n("下一页"))
            with gr.Row():
                helper_add = gr.Button(i18n("新增一行"))
                helper_remove = gr.Button(i18n("删除末行"))
            helper_row_outputs = []
            helper_row_values = []
            for helper_slot in range(MULTISPEAKER_PAGE_SIZE):
                with gr.Row():
                    helper_path = gr.Textbox(
                        label=i18n("训练集子音频文件夹目录路径"),
                        value="",
                        visible=helper_slot < 2,
                    )
                    helper_speaker_name = gr.Textbox(
                        label=i18n("说话人名称"),
                        value="",
                        visible=helper_slot < 2,
                    )
                    helper_speaker_id = gr.Number(
                        label=i18n("说话人ID（0~109）"),
                        value=helper_slot if helper_slot < 2 else None,
                        precision=0,
                        visible=helper_slot < 2,
                    )
                    helper_repeat = gr.Number(
                        label=i18n("重复次数"),
                        value=1 if helper_slot < 2 else None,
                        precision=0,
                        visible=helper_slot < 2,
                    )
                helper_row_outputs.extend(
                    [
                        helper_path,
                        helper_speaker_name,
                        helper_speaker_id,
                        helper_repeat,
                    ]
                )
                helper_row_values.extend(
                    [helper_path, helper_speaker_name, helper_speaker_id, helper_repeat]
                )
            helper_status = gr.HTML(value="")
            helper_event_inputs = [
                helper_rows_state,
                helper_row_count,
                helper_page,
            ] + helper_row_values
            helper_event_outputs = [
                helper_rows_state,
                helper_row_count,
                helper_page,
                helper_page_label,
            ] + helper_row_outputs
            helper_add.click(
                add_multispeaker_row,
                helper_event_inputs,
                helper_event_outputs,
                queue=False,
            )
            helper_remove.click(
                remove_multispeaker_row,
                helper_event_inputs,
                helper_event_outputs,
                queue=False,
            )
            helper_previous.click(
                previous_multispeaker_page,
                helper_event_inputs,
                helper_event_outputs,
                queue=False,
            )
            helper_next.click(
                next_multispeaker_page,
                helper_event_inputs,
                helper_event_outputs,
                queue=False,
            )
            helper_submit.click(
                submit_multispeaker_rows,
                [helper_exp_name] + helper_event_inputs,
                [helper_rows_state, helper_status],
                api_name="multispeaker_manifest_submit",
            )
            exp_dir1.change(
                sync_exp_name,
                [exp_dir1, helper_exp_name],
                [helper_exp_name],
                queue=False,
            )
            helper_exp_name.change(
                sync_exp_name,
                [helper_exp_name, exp_dir1],
                [exp_dir1],
                queue=False,
            )

        with gr.TabItem(i18n("ckpt处理")):
            with gr.Group():
                gr.Markdown(value=i18n("模型融合, 可用于测试音色融合"))
                with gr.Row():
                    ckpt_a = gr.Textbox(
                        label=i18n("A模型路径"), value="", interactive=True
                    )
                    ckpt_b = gr.Textbox(
                        label=i18n("B模型路径"), value="", interactive=True
                    )
                    alpha_a = gr.Slider(
                        minimum=0,
                        maximum=1,
                        label=i18n("A模型权重"),
                        value=0.5,
                        interactive=True,
                    )
                with gr.Row():
                    sr_ = gr.Radio(
                        label=i18n("目标采样率"),
                        choices=["40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0_ = gr.Radio(
                        label=i18n("模型是否带音高指导"),
                        choices=[i18n("是"), i18n("否")],
                        value=i18n("是"),
                        interactive=True,
                    )
                    info__ = gr.Textbox(
                        label=i18n("要置入的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save0 = gr.Textbox(
                        label=i18n("保存的模型名不带后缀"),
                        value="",
                        max_lines=1,
                        interactive=True,
                    )
                    version_2 = gr.Radio(
                        label=i18n("模型版本型号"),
                        choices=["v1", "v2"],
                        value="v1",
                        interactive=True,
                    )
                with gr.Row():
                    but6 = gr.Button(i18n("融合"), variant="primary")
                    info4 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but6.click(
                    merge,
                    [
                        ckpt_a,
                        ckpt_b,
                        alpha_a,
                        sr_,
                        if_f0_,
                        info__,
                        name_to_save0,
                        version_2,
                    ],
                    info4,
                    api_name="ckpt_merge",
                )  # def merge(path1,path2,alpha1,sr,f0,info):
            with gr.Group():
                gr.Markdown(
                    value=i18n("修改模型信息(仅支持weights文件夹下提取的小模型文件)")
                )
                with gr.Row():
                    ckpt_path0 = gr.Textbox(
                        label=i18n("模型路径"), value="", interactive=True
                    )
                    info_ = gr.Textbox(
                        label=i18n("要改的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    name_to_save1 = gr.Textbox(
                        label=i18n("保存的文件名, 默认空为和源文件同名"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                with gr.Row():
                    but7 = gr.Button(i18n("修改"), variant="primary")
                    info5 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but7.click(
                    change_info,
                    [ckpt_path0, info_, name_to_save1],
                    info5,
                    api_name="ckpt_modify",
                )
            with gr.Group():
                gr.Markdown(
                    value=i18n("查看模型信息(仅支持weights文件夹下提取的小模型文件)")
                )
                with gr.Row():
                    ckpt_path1 = gr.Textbox(
                        label=i18n("模型路径"), value="", interactive=True
                    )
                    but8 = gr.Button(i18n("查看"), variant="primary")
                    info6 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                but8.click(show_info, [ckpt_path1], info6, api_name="ckpt_show")
            with gr.Group():
                gr.Markdown(
                    value=i18n(
                        "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况"
                    )
                )
                with gr.Row():
                    ckpt_path2 = gr.Textbox(
                        label=i18n("模型路径"),
                        value="E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth",
                        interactive=True,
                    )
                    save_name = gr.Textbox(
                        label=i18n("保存名"), value="", interactive=True
                    )
                    sr__ = gr.Radio(
                        label=i18n("目标采样率"),
                        choices=["32k", "40k", "48k"],
                        value="40k",
                        interactive=True,
                    )
                    if_f0__ = gr.Radio(
                        label=i18n("模型是否带音高指导,1是0否"),
                        choices=["1", "0"],
                        value="1",
                        interactive=True,
                    )
                    version_1 = gr.Radio(
                        label=i18n("模型版本型号"),
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                    )
                    info___ = gr.Textbox(
                        label=i18n("要置入的模型信息"),
                        value="",
                        max_lines=8,
                        interactive=True,
                    )
                    but9 = gr.Button(i18n("提取"), variant="primary")
                    info7 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    ckpt_path2.change(
                        change_info_, [ckpt_path2], [sr__, if_f0__, version_1]
                    )
                but9.click(
                    extract_small_model,
                    [ckpt_path2, save_name, sr__, if_f0__, info___, version_1],
                    info7,
                    api_name="ckpt_extract",
                )

        with gr.TabItem("วิธีใช้งานภาษาไทย"):
            try:
                guide = read_text("docs/th/quick-start.md")
                gr.Markdown(value=guide)
            except Exception:
                gr.Markdown("เปิดคู่มือภาษาไทยไม่สำเร็จ กรุณาตรวจสอบไฟล์ docs/th/quick-start.md")

        tab_faq = i18n("常见问题解答")
        with gr.TabItem(tab_faq):
            try:
                if tab_faq == "常见问题解答":
                    info = read_text("docs/cn/faq.md")
                else:
                    info = read_text("docs/en/faq_en.md")
                gr.Markdown(value=info)
            except Exception:
                gr.Markdown(traceback.format_exc())

    if config.iscolab:
        app.queue(concurrency_count=511, max_size=1022).launch(share=True)
    else:
        launch_webui_with_port_fallback(app, config)
