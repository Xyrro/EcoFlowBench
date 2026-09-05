#!/usr/bin/env bash
# Re-run the lightweight cluster inspection used to write docs/compute_env.md.
# Safe on a login node (read-only Slurm/queries only). Output goes to stdout.
set -u
echo "### host"; hostname; cat /etc/redhat-release
echo "### user / account / qos"
sacctmgr show user "$USER" withassoc format=User,Account,DefaultAccount,QOS%40
sacctmgr show qos -P format=Name,MaxWall,MaxTRESPU,MaxJobsPU,MaxSubmitPU,GrpTRES,Priority
echo "### partitions"
sinfo -o "%P %a %l %D %t %G %m %c"
scontrol show partition | grep -E "PartitionName|AllowQos|MaxTime|DefMemPerCPU|TRES="
echo "### quota"; pace-quota 2>/dev/null | grep -E "Home|Scratch"
echo "### modules of interest"
for m in julia python anaconda3 miniforge mamba cuda gdal; do echo "-- $m"; module -t avail "$m" 2>&1 | grep -v -E "^$|:$" | head -8; done
