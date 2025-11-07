from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", [0, 1111]),
        ("env", ["doom_pitfall"]),
    ]
)

cpu? = ' --device=cpu'

_experiments = [
    Experiment(
        "doom_marl",
        "python -m sf.train --train_for_env_steps=20000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=8 --num_policies=1 --batch_size=1024 --wide_aspect_ratio=False --with_wandb=True --wandb_dir=. --wandb_record_every=10 --wandb_project=marl_vizdoom",
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription("doom_marl", experiments=_experiments)
# python -m sample_factory.launcher.run --run=sf.train_all --backend=processes --max_parallel=4  --pause_between=1 --experiments_per_gpu=4 --num_gpus=1
# python -m sample_factory.launcher.run --run=sf.train_all --backend=processes --max_parallel=4  --pause_between=1









'''
https://www.samplefactory.dev/04-experiments/experiment-launcher/#local-backend-multiprocessing

Arguments:
-h, --help            show this help message and exit
--train_dir TRAIN_DIR
                        Directory for sub-experiments
--run RUN             Name of the python module that describes the run, e.g.
                        sf_examples.vizdoom.experiments.doom_basic
--backend {processes,slurm,ngc}
--pause_between PAUSE_BETWEEN
                        Pause in seconds between processes
--experiment_suffix EXPERIMENT_SUFFIX
                        Append this to the name of the experiment dir

Multiprocessing backend:
--num_gpus NUM_GPUS   How many GPUs to use (only for local multiprocessing)
--experiments_per_gpu EXPERIMENTS_PER_GPU
                        How many experiments can we squeeze on a single GPU
                        (-1 for not altering CUDA_VISIBLE_DEVICES at all)
--max_parallel MAX_PARALLEL
                        Maximum simultaneous experiments (only for local multiprocessing)

Slurm-related:
--slurm_gpus_per_job SLURM_GPUS_PER_JOB
                        GPUs in a single SLURM process
--slurm_cpus_per_gpu SLURM_CPUS_PER_GPU
                        Max allowed number of CPU cores per allocated GPU
--slurm_print_only SLURM_PRINT_ONLY
                        Just print commands to the console without executing
--slurm_workdir SLURM_WORKDIR
                        Optional workdir. Used by slurm launcher to store
                        logfiles etc.
--slurm_partition SLURM_PARTITION
                        Adds slurm partition, i.e. for "gpu" it will add "-p
                        gpu" to sbatch command line
--slurm_sbatch_template SLURM_SBATCH_TEMPLATE
                        Commands to run before the actual experiment (i.e.
                        activate conda env, etc.) Example: https://github.com/alex-petrenko/megaverse/blob/master/megaverse_rl/slurm/sbatch_template.sh
                        (typically a shell script)
--slurm_timeout SLURM_TIMEOUT
                        Time to run jobs before timing out job and requeuing the job. Defaults to 0, which does not time out the job
'''