
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.utils import str2bool


def add_doom_env_args(parser):
    p = parser

    p.add_argument(
        "--num_agents",
        default=-1,
        type=int,
        help="Allows to set number of agents less than number of players, to allow humans to join the match. Default value (-1) means default number defined by the environment",
    )
    p.add_argument("--num_humans", default=0, type=int, help="Meatbags want to play?")
    p.add_argument(
        "--num_bots",
        default=-1,
        type=int,
        help="Add classic (non-neural) bots to the match. If default (-1) then use number of bots specified in env cfg",
    )
    p.add_argument(
        "--start_bot_difficulty", default=None, type=int, help="Adjust bot difficulty, useful for evaluation"
    )
    p.add_argument(
        "--timelimit", default=None, type=float, help="Allows to override default match timelimit in minutes"
    )
    p.add_argument("--res_w", default=128, type=int, help="Game frame width after resize")
    p.add_argument("--res_h", default=72, type=int, help="Game frame height after resize")
    p.add_argument(
        "--wide_aspect_ratio",
        default=False,
        type=str2bool,
        help="If true render wide aspect ratio (slower but gives better FOV to the agent)",
    )
    p.add_argument(
        "--shared_reward_alpha",
        default=None,
        type=float,
        help="Override the COMRAD joint-env shared reward blend factor after per-agent reward shaping. Applied only for actor-critic multi-agent algos; QMIX/VDN/QPLEX already scalarize rewards once in the learner.",
    )
    p.add_argument(
        "--shared_reward_scalarisation",
        default=None,
        choices=("sum", "mean"),
        help="Override how joint-env shared rewards are scalarized before they are broadcast to actor-critic agents.",
    )
    p.add_argument(
        "--wad_batch",
        default=None,
        type=str,
        help="Path to a batch directory (with batch_registry.json)",
    )
    p.add_argument(
        "--wad_swap_every",
        default=5,
        type=int,
        help="Swap WAD every N episodes (only used with --wad_batch).",
    )
    p.add_argument(
        "--wad_curriculum",
        default="uniform",
        type=str,
        choices=["uniform", "learning_progress", "plr", "sequential", "omni"],
        help="Curriculum strategy for sampling WAD batches. "
            "uniform=equal probability, "
            "learning_progress=prioritize tasks with highest return variance, "
            "plr=prioritize lowest-return tasks, "
            "sequential=advance through tasks in order, "
            "omni=LP masked by interestingness graph.",
    )
    # lp hyperparameters
    p.add_argument(
        "--lp_p_theta",
        default=0.1,
        type=float,
        help="LP curriculum p_theta parameter for rescaling success rates before computing the LP difference. (p_theta=0 => tasks with near-zero success stay unintersting, p_theta=1 => tasks close to full mastery are heavily unprioritized).",
    )
    # omni hyperparameters
    p.add_argument(
        "--interestingness_graph_path",
        default=None,
        type=str,
        help="Path to JSON file containing the interestingness graph for the omni curriculum strategy. "
            "The JSON should be of type dict[str, dict[str, bool]], where the keys of the outer dict are task identifiers (as passed in the batch) and the inner dict maps other task identifiers to booleans indicating whether they remain interesting (True) or become boring (False) once the outer task is mastered.",
    )
    # plr hyperparameters
    p.add_argument(
        "--plr_replay_schedule",
        default="proportionate",
        type=str,
        choices=["proportionate", "fixed"],
        help="PLR replay schedule."
    )
    p.add_argument(
        "--plr_replay_prob",
        default=0.5,
        type=float,
        help="PLR probability of sampling a replay level."
    )
    p.add_argument(
        "--plr_rho",
        default=1.0,
        type=float,
        help="PLR proportion of tasks that must be seen before replay starts."
    )
    p.add_argument(
        "--plr_staleness_coef",
        default=0.1,
        type=float,
        help="PLR staleness interpolation coefficient."
    )
    p.add_argument(
        "--plr_score_transform",
        default="rank",
        type=str,
        choices=["rank", "power", "softmax", "max", "constant"],
        help="PLR score transform."
        )
    p.add_argument(
        "--plr_temperature",
        default=0.1,
        type=float,
        help="PLR score transform temperature."
    )
    p.add_argument(
        "--plr_alpha",
        default=1.0,
        type=float,
        help="PLR score interpolation weight (1.0 = use new score only)."
        )
    p.add_argument(
        "--plr_staleness_transform",
        default="power",
        type=str,
        help="PLR staleness transform."
    )
    p.add_argument(
        "--plr_staleness_temperature",
        default=1.0,
        type=float,
        help="PLR staleness transform temperature."
    )
    p.add_argument(
        "--plr_score_key",
        default="mean_value_l1",
        type=str,
        choices=["mean_value_l1", "mean_advantage", "mean_entropy"],
        help="PLR score metric (mean signal) used from learner scoring. Max is derived automatically."
    )
    p.add_argument(
        "--plr_max_score_coef",
        default=0.0,
        type=float,
        help="PLR interpolation weight between max score and mean score (0.0 = mean only)."
    )
    p.add_argument(
        "--plr_eps",
        default=0.05,
        type=float,
        help="PLR epsilon for minimum replay probability."
    )


def add_wandb_args(parser):
    parser.add_argument("--wandb_record_every", default=50, type=int, help="Every N episodes")
    parser.add_argument("--wandb_video_fps", default=35, type=int)


def add_doom_env_eval_args(parser):
    """Arguments used only during evaluation."""
    parser.add_argument(
        "--record_to",
        # default=join(os.getcwd(), "..", "recs"),
        default=None,
        type=str,
        help="Record episodes to this folder. This records a demo that can be replayed at full resolution. Currently, this does not work for bot environments so it is recommended to use --save_video to record episodes at lower resolution instead for such environments",
    )


def doom_override_defaults(parser):
    """RL params specific to Doom envs."""
    parser.set_defaults(
        ppo_clip_value=0.2,  # value used in all experiments in the paper
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        exploration_loss="symmetric_kl",
        exploration_loss_coeff=0.001,
        normalize_returns=True,
        normalize_input=True,
        env_frameskip=4,
        eval_env_frameskip=1,  # this is for smoother rendering during evaluation
        fps=35,  # for evaluation only
        heartbeat_reporting_interval=600,
    )


def default_doom_cfg(algo="APPO", env="env", experiment="test"):
    """Useful in tests."""
    argv = [f"--algo={algo}", f"--env={env}", f"--experiment={experiment}"]
    parser, args = parse_sf_args(argv)
    add_doom_env_args(parser)
    doom_override_defaults(parser)
    args = parse_full_cfg(parser, argv)
    return args
