#!/bin/bash

# Reassemble the latent_balancer.pth file from the parts
cat latent_balancer_part_* > latent_balancer.pth

echo "Weights have been reassembled into latent_balancer.pth."

