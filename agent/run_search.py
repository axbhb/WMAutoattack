from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from agent.llm import (
    HeuristicAttackerAgent,
    HeuristicAuditorAgent,
    OpenAIAttackerAgent,
    OpenAIAuditorAgent,
    TransformersAttackerAgent,
    TransformersAuditorAgent,
    TransformersResponsesClient,
)
from agent.orchestration import DebateSearchController
from agent.schema import AttackSearchSpace, SearchConfig, TaskSpec
from autoattack.runtime import DreamerV3SearchExecutor


def default_search_spaces() -> Dict[str, AttackSearchSpace]:
    return {
        "apgd_ce": AttackSearchSpace(
            attack_name="apgd_ce",
            epsilons=(4, 6, 8, 10, 12),
            step_candidates=(4, 6, 8, 10, 12),
            allocation_modes=("fixed", "margin_linear"),
        ),
        "apgd_dlr": AttackSearchSpace(
            attack_name="apgd_dlr",
            epsilons=(4, 6, 8, 10, 12),
            step_candidates=(4, 6, 8, 10, 12),
            allocation_modes=("fixed", "margin_linear"),
        ),
        "fab": AttackSearchSpace(
            attack_name="fab",
            epsilons=(4, 6, 8, 10, 12),
            step_candidates=(6, 8, 10, 12, 16),
            allocation_modes=("fixed", "margin_linear"),
        ),
        "two_stage": AttackSearchSpace(
            attack_name="two_stage",
            epsilons=(4, 6, 8, 10, 12),
            step_candidates=(6, 8, 10, 12, 16),
            allocation_modes=("fixed", "margin_linear"),
        ),
        "square": AttackSearchSpace(
            attack_name="square",
            epsilons=(4, 6, 8, 10),
            step_candidates=(20, 40, 60, 80),
            allocation_modes=("fixed", "margin_linear"),
            min_step_fractions=(0.25, 0.4, 0.6),
        ),
    }


def discover_tasks(args: argparse.Namespace) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    if args.checkpoint_path:
        for checkpoint in args.checkpoint_path:
            checkpoint_path = Path(checkpoint).resolve()
            tasks.append(TaskSpec(name=checkpoint_path.parent.parent.parent.parent.name, checkpoint_path=str(checkpoint_path)))
        return tasks

    if not args.task_root:
        raise ValueError("You must specify either --checkpoint-path or --task-root.")

    root = Path(args.task_root).resolve()
    games = [game.strip() for game in args.games.split(",")] if args.games else None
    game_dirs: Iterable[Path]
    if games:
        game_dirs = [root / game for game in games]
    else:
        game_dirs = [path for path in root.iterdir() if path.is_dir()]

    for game_dir in game_dirs:
        matches = sorted(game_dir.rglob(args.checkpoint_name))
        if len(matches) == 0:
            continue
        tasks.append(TaskSpec(name=game_dir.name, checkpoint_path=str(matches[0])))

    if len(tasks) == 0:
        raise FileNotFoundError("No checkpoint matching the given search criteria was found.")
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-agent search for DreamerV3 AutoAttack parameters.")
    parser.add_argument("--task-root", type=str, default="", help="Root directory that contains game folders.")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        nargs="*",
        default=[],
        help="One or more explicit checkpoint paths. Overrides --task-root discovery.",
    )
    parser.add_argument("--games", type=str, default="", help="Comma-separated list of game folder names.")
    parser.add_argument("--checkpoint-name", type=str, default="ckpt_100000_0.ckpt")
    parser.add_argument("--attacks", type=str, default="apgd_ce,apgd_dlr,fab,two_stage,square")
    parser.add_argument("--scout-episodes", type=int, default=3)
    parser.add_argument("--confirm-episodes", type=int, default=10)
    parser.add_argument("--proposal-batch-size", type=int, default=4)
    parser.add_argument("--confirm-top-k", type=int, default=2)
    parser.add_argument("--max-trials-per-attack", type=int, default=6)
    parser.add_argument("--runtime-budget-seconds", type=float, default=600.0)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--agent-backend", choices=("transformers", "openai", "heuristic"), default="transformers")
    parser.add_argument("--agent-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--agent-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--agent-base-url", type=str, default="")
    parser.add_argument("--agent-max-candidates", type=int, default=24)
    parser.add_argument("--agent-context-limit", type=int, default=8)
    parser.add_argument("--agent-max-new-tokens", type=int, default=384)
    parser.add_argument("--agent-temperature", type=float, default=0.0)
    parser.add_argument("--agent-cache-dir", type=str, default="")
    parser.add_argument("--experience-store-path", type=str, default="logs/agent_experience.jsonl")
    parser.add_argument("--experience-retrieval-limit", type=int, default=6)
    parser.add_argument(
        "--experience-retrieval-mode",
        choices=("structured", "latent", "hybrid"),
        default="structured",
        help="Experience retrieval backend. 'structured' preserves the existing field-based RAG.",
    )
    parser.add_argument(
        "--experience-latent-projection",
        choices=("pca", "identity"),
        default="pca",
        help="How to compress pooled probe features into a latent retrieval space.",
    )
    parser.add_argument("--experience-latent-dim", type=int, default=16)
    parser.add_argument("--experience-hybrid-weight", type=float, default=0.6)
    parser.add_argument("--experience-probe-max-steps", type=int, default=32)
    return parser


def build_agents(search_config: SearchConfig):
    if search_config.agent_backend == "heuristic":
        return HeuristicAttackerAgent(), HeuristicAuditorAgent()

    if search_config.agent_backend == "transformers":
        client = TransformersResponsesClient(
            cache_dir=search_config.agent_cache_dir,
            max_new_tokens=search_config.agent_max_new_tokens,
            temperature=search_config.agent_temperature,
        )
        return (
            TransformersAttackerAgent(
                client=client,
                model=search_config.agent_model,
                max_candidates=search_config.agent_max_candidates,
                context_limit=search_config.agent_context_limit,
            ),
            TransformersAuditorAgent(
                client=client,
                model=search_config.agent_model,
                context_limit=search_config.agent_context_limit,
            ),
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI-backed agents require the 'openai' package. Install it with `pip install openai` "
            "or `pip install -e .[llm]`."
        ) from exc

    api_key = os.getenv(search_config.agent_api_key_env, "")
    if not api_key:
        raise RuntimeError(
            "OpenAI-backed agents require an API key in the environment variable '{}'.".format(
                search_config.agent_api_key_env
            )
        )

    client_kwargs = {"api_key": api_key}
    if search_config.agent_base_url:
        client_kwargs["base_url"] = search_config.agent_base_url
    client = OpenAI(**client_kwargs)
    return (
        OpenAIAttackerAgent(
            client=client,
            model=search_config.agent_model,
            max_candidates=search_config.agent_max_candidates,
            context_limit=search_config.agent_context_limit,
        ),
        OpenAIAuditorAgent(
            client=client,
            model=search_config.agent_model,
            context_limit=search_config.agent_context_limit,
        ),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = args.output_dir or str(
        Path("logs")
        / "attack_search"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    search_config = SearchConfig(
        output_dir=output_dir,
        scout_episodes=args.scout_episodes,
        confirm_episodes=args.confirm_episodes,
        proposal_batch_size=args.proposal_batch_size,
        confirm_top_k=args.confirm_top_k,
        max_trials_per_attack=args.max_trials_per_attack,
        runtime_budget_seconds=args.runtime_budget_seconds,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        seed=args.seed,
        agent_backend=args.agent_backend,
        agent_model=args.agent_model,
        agent_api_key_env=args.agent_api_key_env,
        agent_base_url=args.agent_base_url,
        agent_max_candidates=args.agent_max_candidates,
        agent_context_limit=args.agent_context_limit,
        agent_max_new_tokens=args.agent_max_new_tokens,
        agent_temperature=args.agent_temperature,
        agent_cache_dir=args.agent_cache_dir,
        experience_store_path=args.experience_store_path,
        experience_retrieval_limit=args.experience_retrieval_limit,
        experience_retrieval_mode=args.experience_retrieval_mode,
        experience_latent_projection=args.experience_latent_projection,
        experience_latent_dim=args.experience_latent_dim,
        experience_hybrid_weight=args.experience_hybrid_weight,
        experience_probe_max_steps=args.experience_probe_max_steps,
    )
    tasks = discover_tasks(args)
    spaces = default_search_spaces()
    enabled_attacks = {attack.strip() for attack in args.attacks.split(",") if attack.strip()}
    spaces = {name: space for name, space in spaces.items() if name in enabled_attacks}
    if len(spaces) == 0:
        raise ValueError("No valid attack search spaces remain after filtering --attacks.")

    executor = DreamerV3SearchExecutor(search_config)
    attacker_agent, auditor_agent = build_agents(search_config)
    controller = DebateSearchController(
        executor=executor,
        attacker_agent=attacker_agent,
        auditor_agent=auditor_agent,
        search_config=search_config,
    )
    summary = controller.run(tasks, spaces)
    print(f"Search finished. Summary written to: {Path(search_config.output_dir) / 'search_summary.json'}")
    print(f"Processed tasks: {[task['task'] for task in summary['tasks']]}")


if __name__ == "__main__":
    main()
