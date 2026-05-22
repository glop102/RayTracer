## Building
We are a nix-first dev, so we use nix build to build things

nix build .#raytracer -L 2>&1
nix build .#raytracer_vk -L 2>&1
