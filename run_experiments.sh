#!/usr/bin/env bash

for ((i=0;i<15;i+=1))
do
    docker run --detach \
     -v $(pwd):/root/work/domain-randomization -it sharadmv/domain-randomization \
     pipenv run python scripts/ppo.py  --seed $i --collision_detector ode --env_dist_stdev 0.5 \
     --env_name Walker
done
