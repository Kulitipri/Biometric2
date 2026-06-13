"""Face restoration (CodeFormer) — tiền xử lý phá trần cho mặt phân giải thấp.

Lever A (break-ceiling): phục chế mặt nhỏ TRƯỚC khi detect+align, đánh vào nút
thắt resolution của RMFRD (median short side 107px). Recognition vẫn pretrained
— đây chỉ là tầng tiền xử lý generative, KHÔNG train lại embedder.

Thiết kế: bỏ qua face_helper/aligner nội bộ của CodeFormer (ảnh RMFRD đã là crop
mặt nhỏ, gần vuông ~150x140, aligner dễ fail). Dùng thẳng restoration net:
resize crop → 512x512 → CodeFormer → ảnh phục chế 512x512; alignment vẫn do
RetinaFace của pipeline đảm nhiệm trên ảnh đã phục chế.

Giả định: crop RMFRD gần vuông nên resize 512x512 chỉ méo nhẹ (~7%). Với ảnh
non-square mạnh thì nên align trước rồi mới restore (chưa cần cho RMFRD).

Fidelity weight (w): w→0 ưu tiên chất lượng (hallucinate nhiều, rủi ro đổi
identity); w→1 trung thực với input (an toàn identity). Mặc định 0.7.

Gate: chỉ restore mặt có short side < min_short_side; mặt đủ rõ trả nguyên bản
(zero-overhead) để không làm hại ảnh tốt.
"""

from __future__ import annotations

import cv2
import numpy as np

# Hyperparams kiến trúc CodeFormer (cố định theo checkpoint chính thức).
_CODEFORMER_ARCH = dict(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=["32", "64", "128", "256"],
)


class FaceRestorer:
    """Wrap CodeFormer restoration net với gate theo kích thước."""

    def __init__(
        self,
        weights: str = "weights/codeformer.pth",
        min_short_side: int = 100,
        fidelity: float = 0.7,
        device: str = "cuda",
    ) -> None:
        self.weights = weights
        self.min_short_side = int(min_short_side)
        self.fidelity = float(fidelity)
        self.device = device
        self._net = None  # lazy: chỉ load khi thực sự có ảnh cần restore
        self._torch = None

    def _ensure_model(self) -> None:
        """Lazy-load CodeFormer net + weights lần đầu cần dùng (idempotent)."""
        if self._net is not None:
            return
        import torch

        try:
            from basicsr.utils.registry import ARCH_REGISTRY
            # Import side-effect để register arch 'CodeFormer'. Thử cả 2 nguồn:
            # codeformer-pip (package 'codeformer') hoặc arch đã có trong basicsr.
            try:
                import codeformer.basicsr.archs.codeformer_arch  # noqa: F401
            except Exception:
                from basicsr.archs import codeformer_arch  # noqa: F401
            net = ARCH_REGISTRY.get("CodeFormer")(**_CODEFORMER_ARCH)
        except Exception as e:
            raise ImportError(
                "Không load được kiến trúc CodeFormer. Cài đặt một trong hai:\n"
                "  pip install codeformer-pip\n"
                "hoặc vendor repo (giống third_party/MaskTheFace):\n"
                "  git clone https://github.com/sczhou/CodeFormer third_party/CodeFormer\n"
                "  pip install basicsr facexlib\n"
                "Lỗi basicsr 'functional_tensor' (torchvision mới): sửa import trong\n"
                "  basicsr/data/degradations.py: 'torchvision.transforms.functional_tensor'\n"
                "  → 'torchvision.transforms.functional'.\n"
                f"Chi tiết: {e}"
            ) from e

        ckpt = torch.load(self.weights, map_location="cpu")
        state = ckpt.get("params_ema", ckpt.get("params", ckpt)) if isinstance(ckpt, dict) else ckpt
        net.load_state_dict(state)
        net.eval()
        self._net = net.to(self.device)
        self._torch = torch

    def restore(self, img_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """Phục chế 1 ảnh BGR nếu mặt nhỏ; trả (ảnh_kết_quả, đã_restore?).

        Mặt đủ to (short side >= min_short_side) → trả (img_bgr, False) ngay,
        không động vào model. Ngược lại → resize 512 → CodeFormer → 512x512 BGR.
        """
        if min(img_bgr.shape[:2]) >= self.min_short_side:
            return img_bgr, False
        self._ensure_model()
        torch = self._torch

        # BGR→RGB, resize 512x512, normalize [-1,1], NCHW tensor.
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        x = inp.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self._net(x, w=self.fidelity, adain=True)[0]

        # Tensor [-1,1] → BGR uint8 512x512.
        out = out.squeeze(0).clamp_(-1, 1).cpu().numpy()
        out = (out.transpose(1, 2, 0) + 1.0) / 2.0 * 255.0
        restored_bgr = cv2.cvtColor(out.round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        return restored_bgr, True


def build_restorer(enabled: bool = False, **kwargs) -> "FaceRestorer | None":
    """Factory: trả None khi tắt (pipeline bỏ qua hẳn restoration)."""
    if not enabled:
        return None
    # Lọc các key không phải tham số FaceRestorer (vd 'enabled' đã tách).
    allowed = {"weights", "min_short_side", "fidelity", "device"}
    return FaceRestorer(**{k: v for k, v in kwargs.items() if k in allowed})
