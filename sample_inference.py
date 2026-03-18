from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single X-VLA forward pass with LIBERO defaults."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="2toINF/X-VLA-Libero",
        help="Local checkpoint directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        help="Optional processor path. Defaults to --model_path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for inference.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="pick up the object and place it in the box",
        help="Language instruction sent to the policy.",
    )
    parser.add_argument(
        "--image0",
        type=str,
        default=None,
        help="Path to the main LIBERO camera image.",
    )
    parser.add_argument(
        "--image1",
        type=str,
        default=None,
        help="Path to the wrist camera image.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Fallback blank image size when no image path is provided.",
    )
    parser.add_argument(
        "--proprio",
        type=str,
        default=None,
        help="Optional comma-separated proprio vector with 10 or 20 floats.",
    )
    parser.add_argument(
        "--proprio_path",
        type=str,
        default=None,
        help="Optional .npy file containing a 10-D or 20-D proprio vector.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of denoising steps passed to generate_actions().",
    )
    parser.add_argument(
        "--domain_id",
        type=int,
        default=3,
        help="Domain id. LIBERO uses 3 in this repo.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the generated action plan as .npy.",
    )
    parser.add_argument(
        "--print_all",
        action="store_true",
        help="Print the full action plan instead of only summary rows.",
    )
    parser.add_argument(
        "--random_dummy_inputs",
        action="store_true",
        help="Generate non-zero random LIBERO-style dummy images and proprio.",
    )
    parser.add_argument(
        "--dummy_seed",
        type=int,
        default=7,
        help="Seed used when --random_dummy_inputs is enabled.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_images(args: argparse.Namespace, rng: np.random.Generator | None) -> list[Image.Image]:
    images: list[Image.Image] = []
    if args.random_dummy_inputs and rng is not None:
        for _ in range(2):
            image = rng.integers(
                1,
                256,
                size=(args.image_size, args.image_size, 3),
                dtype=np.uint8,
            )
            images.append(Image.fromarray(image))
        return images

    for image_path in (args.image0, args.image1):
        if image_path is None:
            continue
        image = Image.open(image_path).convert("RGB")
        images.append(image)

    if not images:
        blank = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
        images = [Image.fromarray(blank)]

    return images


def parse_csv_proprio(raw: str) -> np.ndarray:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    return np.asarray(values, dtype=np.float32)


def normalize_proprio(proprio: np.ndarray) -> np.ndarray:
    proprio = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if proprio.shape[0] == 20:
        return proprio
    if proprio.shape[0] == 10:
        return np.concatenate([proprio, np.zeros_like(proprio)], axis=0)
    raise ValueError(
        f"Expected a 10-D or 20-D proprio vector, but got shape {proprio.shape}."
    )


def load_proprio(args: argparse.Namespace, rng: np.random.Generator | None) -> np.ndarray:
    if args.proprio and args.proprio_path:
        raise ValueError("Use only one of --proprio or --proprio_path.")

    if args.random_dummy_inputs and rng is not None and not args.proprio and not args.proprio_path:
        proprio = rng.normal(loc=0.1, scale=0.25, size=(10,)).astype(np.float32)
    elif args.proprio_path:
        proprio = np.load(args.proprio_path)
    elif args.proprio:
        proprio = parse_csv_proprio(args.proprio)
    else:
        proprio = np.zeros(20, dtype=np.float32)

    return normalize_proprio(proprio)


def rotate6d_to_matrix(rotate6d: np.ndarray) -> np.ndarray:
    a1 = rotate6d[..., 0:3]
    a2 = rotate6d[..., 3:6]

    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), EPS, None)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - proj * b1
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), EPS, None)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    axis = np.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        axis=-1,
    )
    axis = axis / np.clip(2.0 * np.sin(theta)[..., None], EPS, None)
    axis_angle = axis * theta[..., None]
    axis_angle[theta < 1e-4] = 0.0
    return axis_angle.astype(np.float32)


def libero_action_to_axis_angle(actions: np.ndarray) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[-1] < 10:
        raise ValueError(f"Expected [T, 10+] actions, got {actions.shape}.")
    rotation = rotate6d_to_matrix(actions[:, 3:9])
    axis_angle = matrix_to_axis_angle(rotation)
    return np.concatenate([actions[:, :3], axis_angle, actions[:, 9:10]], axis=-1)


def main() -> None:
    args = parse_args()

    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    device = resolve_device(args.device)
    rng = np.random.default_rng(args.dummy_seed) if args.random_dummy_inputs else None

    processor_path = args.processor_path or args.model_path
    model = XVLA.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device).to(torch.float32)
    processor = XVLAProcessor.from_pretrained(processor_path)

    images = load_images(args, rng)
    proprio = load_proprio(args, rng)
    inputs = processor(images=images, language_instruction=args.instruction)

    model_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                model_inputs[key] = value.to(device=device, dtype=torch.float32)
            else:
                model_inputs[key] = value.to(device=device)

    proprio_tensor = torch.as_tensor(proprio, dtype=torch.float32, device=device).unsqueeze(0)
    domain_id = torch.tensor([args.domain_id], dtype=torch.long, device=device)

    with torch.inference_mode():
        actions = model.generate_actions(
            input_ids=model_inputs["input_ids"],
            image_input=model_inputs["image_input"],
            image_mask=model_inputs["image_mask"],
            domain_id=domain_id,
            proprio=proprio_tensor,
            steps=args.steps,
        )

    actions_np = actions.squeeze(0).detach().cpu().numpy().astype(np.float32)
    print(f"model_path: {args.model_path}")
    print(f"device: {device}")
    print(f"num_views: {len(images)}")
    print(f"domain_id: {args.domain_id}")
    print(f"proprio_shape: {tuple(proprio.shape)}")
    print(f"action shape: {actions_np.shape}")
    print("first action:", np.array2string(actions_np[0], precision=4))
    if actions_np.shape[0] > 1:
        print("last action:", np.array2string(actions_np[-1], precision=4))

    if actions_np.shape[1] >= 10:
        axis_angle_actions = libero_action_to_axis_angle(actions_np[:, :10])
        print(
            "first action (xyz + axis-angle + gripper):",
            np.array2string(axis_angle_actions[0], precision=4),
        )

    if args.print_all:
        print("full action plan:")
        print(np.array2string(actions_np, precision=4))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, actions_np)
        print(f"saved actions to: {output_path}")


if __name__ == "__main__":
    main()
