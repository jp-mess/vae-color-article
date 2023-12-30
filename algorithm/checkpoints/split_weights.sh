#!/bin/bash

# Split the latent_balancer.pth file into two parts
split -n 2 -d latent_balancer.pth latent_balancer_part_

echo "Weights have been split into two parts."

