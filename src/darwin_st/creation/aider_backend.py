"""Aider 后端 (Aider Backend) —— 用 aider CLI 在 git 沙箱里写算子代码。

替换"LLM 直生代码 + 正则抽取 + exec"那几步(脆弱), 改用成熟的 aider:
  - aider 用 SEARCH/REPLACE diff 直接编辑文件并自动应用(健壮, 不靠正则抽取)
  - 在【临时 git worktree 沙箱】里写, 不污染主代码库
  - 非交互: --yes --no-auto-commits --message
  - 接 DeepSeek: --model deepseek/<model>, DEEPSEEK_API_KEY 从 env

write_operator(instruction, class_name) → (code, error):
  在沙箱里让 aider 把一个 nn.Module 类写进 op.py, 读回代码。失败返回 error。

设计: 可被 synthesizer 注入(真实 AiderBackend / 测试用 mock)。不依赖真 LLM 即可
单元测试本模块的【沙箱管理逻辑】(aider 调用本身需服务器实跑验收)。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

__all__ = ["AiderConfig", "AiderBackend"]


@dataclass
class AiderConfig:
    model: str = "deepseek/deepseek-chat"   # aider 的 DeepSeek 模型标识
    timeout_s: float = 180.0
    aider_bin: str = "aider"
    extra_args: tuple[str, ...] = ()


class AiderBackend:
    """在临时 git 沙箱里用 aider 写算子。"""

    def __init__(self, config: AiderConfig | None = None):
        self.cfg = config or AiderConfig()

    def write_operator(self, instruction: str, target_file: str = "op.py") -> tuple[str | None, str]:
        """在隔离 git 沙箱里让 aider 按 instruction 写 target_file, 返回 (代码, 错误)。

        成功: (代码字符串, "")。失败: (None, 错误信息)。
        沙箱用完即删, 绝不碰主代码库。
        """
        if not os.environ.get("DEEPSEEK_API_KEY"):
            return None, "DEEPSEEK_API_KEY 未设置"

        sandbox = tempfile.mkdtemp(prefix="darwin_synth_")
        try:
            return self._run_in_sandbox(sandbox, instruction, target_file)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    def _run_in_sandbox(self, sandbox: str, instruction: str, target_file: str) -> tuple[str | None, str]:
        fpath = os.path.join(sandbox, target_file)
        # 初始化 git 沙箱 (aider 需要 git repo)
        for cmd in (["git", "init", "-q"],
                    ["git", "config", "user.email", "synth@darwin.st"],
                    ["git", "config", "user.name", "darwin-synth"]):
            subprocess.run(cmd, cwd=sandbox, capture_output=True)
        with open(fpath, "w") as f:
            f.write("# placeholder for synthesized operator\n")
        subprocess.run(["git", "add", "-A"], cwd=sandbox, capture_output=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=sandbox, capture_output=True)

        cmd = [
            self.cfg.aider_bin, "--model", self.cfg.model,
            "--yes", "--no-auto-commits", "--no-stream", "--no-gitignore",
            "--no-check-update", "--no-show-model-warnings",
            *self.cfg.extra_args,
            "--message", instruction, target_file,
        ]
        try:
            proc = subprocess.run(cmd, cwd=sandbox, capture_output=True, text=True,
                                  timeout=self.cfg.timeout_s)
        except subprocess.TimeoutExpired:
            return None, f"aider 超时 ({self.cfg.timeout_s}s)"
        except FileNotFoundError:
            return None, f"找不到 aider 可执行 ({self.cfg.aider_bin})"

        if not os.path.exists(fpath):
            return None, f"aider 未产出文件; stderr: {proc.stderr[:300]}"
        with open(fpath) as f:
            code = f.read()
        if "class " not in code:
            return None, f"产出无类定义; aider 输出: {proc.stdout[-300:]}"
        return code, ""
