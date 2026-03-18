import argparse
import json
import logging
import types
from pathlib import Path

import torch
from diffusers import FluxPipeline
from tqdm import tqdm

from forwards import tc_pade_flux_forward
from utils import ResidualPredictor

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLUX image generation with TC-Pade acceleration")
    parser.add_argument("--model_path", type=str, default="path_to_flux.1-dev", help="Path to the FLUX model")
    parser.add_argument("--prompts_file", type=str, default="./example_prompts.json", help="Path to prompts JSON file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if not set)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_predict", action="store_true", help="Enable TC-Pade residual prediction")
    parser.add_argument("--start_step", type=int, default=4, help="Start step for prediction")
    parser.add_argument("--interval", type=int, default=8, help="Prediction interval")
    parser.add_argument("--N", type=float, default=1.4, help="Curvature threshold (larger N for faster inference)")
    parser.add_argument("--predictor_order", type=int, default=3, help="Predictor order")
    parser.add_argument("--predictor_history_size", type=int, default=6, help="Predictor history size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    return parser.parse_args()


def load_prompts(prompts_file: str) -> list:
    with open(prompts_file, "r") as f:
        return json.load(f)["prompts"]


def build_pipeline(args: argparse.Namespace) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

    transformer = pipe.transformer
    transformer.current_step = {"step": 0}
    transformer.use_predict = args.use_predict
    transformer.num_steps = args.num_inference_steps
    transformer.forward = types.MethodType(tc_pade_flux_forward, transformer)
    transformer.startstep = args.start_step
    transformer.endstep = args.num_inference_steps - 1
    transformer.interval = args.interval
    transformer.residual_predictor = ResidualPredictor(
        order=args.predictor_order,
        history_size=args.predictor_history_size,
        N=args.N,
        device=args.device,
        dtype=torch.bfloat16,
    )

    pipe.to(args.device)
    return pipe


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)

    if args.use_predict:
        name = f"carp_step-{args.num_inference_steps}_N-{args.N}_interval-{args.interval}"
    else:
        name = f"ori_step-{args.num_inference_steps}"
    return Path("./test") / name


def run_inference(pipe: FluxPipeline, prompts: list, args: argparse.Namespace, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_peak_memory = torch.cuda.max_memory_allocated(device=args.device)
    torch.cuda.reset_peak_memory_stats()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for idx, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        pipe.transformer.forward = types.MethodType(tc_pade_flux_forward, pipe.transformer)
        pipe.transformer.current_step["step"] = 0

        start_event.record()
        image = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0]
        end_event.record()
        torch.cuda.synchronize()

        elapsed = start_event.elapsed_time(end_event) * 1e-3
        peak_memory = torch.cuda.max_memory_allocated(device=args.device)
        image.save(output_dir / f"{idx:04d}.png")

        logger.info(
            "time: %.2f s | param memory: %.2f GB | peak memory: %.2f GB",
            elapsed,
            parameter_peak_memory / 1e9,
            peak_memory / 1e9,
        )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    prompts = load_prompts(args.prompts_file)
    logger.info("Loaded %d prompts from %s", len(prompts), args.prompts_file)

    pipe = build_pipeline(args)
    output_dir = resolve_output_dir(args)
    logger.info("Output directory: %s", output_dir)

    run_inference(pipe, prompts, args, output_dir)


if __name__ == "__main__":
    main()
