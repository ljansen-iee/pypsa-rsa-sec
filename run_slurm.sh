# SLURM specifications made in default.cluster.yaml & the individual rules
#snakemake --cluster-config config/config.cluster.yaml --cluster "sbatch -p {cluster.partition} -t {cluster.walltime} -o {cluster.output} -e {cluster.error} -c {threads} --mem {resources.mem_mb}" --jobs 199 --latency-wait 60 --keep-going --configfile config/config.yaml
snakemake solve_sector_scenario_matrix --cluster-config config/config.cluster.yaml --cluster "sbatch -p {cluster.partition} -t {cluster.walltime} -c {cluster.cpus_per_task} --mem {cluster.mem_mb} -x {cluster.exclude}" --jobs 199 --latency-wait 60 --keep-going
